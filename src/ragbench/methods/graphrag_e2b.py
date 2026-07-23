"""GraphRAG (E2B 계열) — 산문 추출 그래프 + 단계적 검색 개선.

추출 아이디어: 작은 모델(gemma E2B 등)은 엄격한 JSON 구조화 추출엔 실패하지만,
자유 산문 "이름 | 유형 | 설명" 형식은 잘 만든다. 그 산문을 결정론적으로 파싱해
EntityNode(설명 포함) + Relation 으로 변환한다. Gemini 등 강한 모델로 교체하면
description 채움률·타입 정확도가 오른다(추출 LLM은 config로 교체).

기법 계열(같은 그래프, 검색만 다름 — registry.py에서 선택):
- GraphRAGE2B         : 그래프 검색 + E5Rerank 재순위 (기준)
- GraphRAGE2BL5       : 위 + 커뮤니티 요약 주입(전역 강화)
- GraphRAGE2BAdaptive : 위 + 질의 유형 라우팅(global→요약 / specific→재순위만)
- GraphRAGE2BHybrid   : ★ 그래프 검색 + 직접 청크 벡터검색 RRF 융합

검증 결론(CLAUDE.md §10.5): 그래프의 저성능은 방법론의 구조적 한계가 아니라
① E5Rerank 임베딩 버그(아래 clean() 참고) + ② 단일 검색 기법이었다. Hybrid로
그래프+벡터를 융합하니 judge 0.333→0.806으로 평면 정밀도를 따라잡았다.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
from typing import Any, Sequence

from .graphrag import GraphRAG

# 장기 인덱싱 견고화 (2026-07-22 행(hang) 사고 대응)
# 공급자(Gemini)가 502/503 을 뱉으면 재시도 중 연결이 물려 **타임아웃 없이 무한 대기**했고,
# 체크포인트가 없어 2h42m 작업의 중간 결과가 0이었다. 아래 셋으로 같은 손실을 막는다.
_EXTRACT_TIMEOUT_S = 120.0   # 청크당 LLM 호출 상한. 넘으면 그 청크만 포기하고 진행.
_EXTRACT_RETRIES = 3         # 타임아웃·일시장애(502/503) 재시도 횟수
_PROGRESS_EVERY = 50         # 진행 로그 간격(청크). stderr 로 즉시 flush.


def _log(msg: str) -> None:
    """진행/경고를 stderr 로 **즉시 flush** 해서 출력한다.

    tqdm 진행바는 버퍼에 갇혀 파이프 뒤에서 안 보인다 — 행(hang) 감지가 늦어진
    직접 원인이었다. 이 로그는 라인 단위로 바로 나간다.

    입력: msg — 출력할 한 줄
    출력: 없음(stderr 출력)
    """
    # 앞의 \n: tqdm 진행바가 \r 로 같은 줄을 물고 있어, 붙이면 로그가 섞여 안 보인다.
    print(f"\n[extract] {msg}", file=sys.stderr, flush=True)


def _cache_key(text: str) -> str:
    """청크 본문으로 체크포인트 키를 만든다(본문이 같으면 추출 결과도 같다).

    입력: text — 추출에 넣는 청크 본문
    출력: sha1 16자리 키
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class _ExtractCache:
    """추출 체크포인트 — 청크별 결과를 JSONL 로 append 해 중단 시 재개를 가능하게 한다.

    왜: 2026-07-22 인덱싱이 2h42m 만에 행(hang)했는데 중간 결과가 **0** 이었다.
    LLM 호출 1건이 끝날 때마다 즉시 디스크에 붙이므로, 언제 죽어도 그때까지가 남는다.
    스레드/코루틴이 섞이므로 파일 쓰기는 락으로 직렬화한다.
    """

    def __init__(self) -> None:
        """빈 캐시 상태로 초기화한다(open 전까지 아무 것도 하지 않음)."""
        self._data: dict[str, tuple[list, list]] = {}
        self._fh = None
        self._lock = threading.Lock()
        self._done = 0
        self._total = 0
        self._t0 = 0.0

    def open(self, path: str | None, total: int) -> None:
        """기존 체크포인트를 읽어들이고 append 핸들을 연다.

        입력: path — JSONL 경로(None 이면 캐시 비활성) / total — 전체 청크 수
        출력: 없음(재개 가능 건수를 로그로 알림)
        """
        self._data, self._done, self._total, self._t0 = {}, 0, total, time.time()
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 죽는 순간 잘린 마지막 줄은 버린다
                    self._data[rec["k"]] = (rec["e"], rec["r"])
        self._fh = open(path, "a", encoding="utf-8")
        if self._data:
            _log(f"체크포인트 {len(self._data)}건 재사용 → 남은 {max(total - len(self._data), 0)}건만 호출")

    def get(self, key: str) -> tuple[list, list] | None:
        """캐시된 추출 결과를 돌려준다(없으면 None)."""
        return self._data.get(key)

    def put(self, key: str, ents: list, rels: list) -> None:
        """추출 결과를 메모리와 디스크(JSONL)에 즉시 기록한다."""
        self._data[key] = (ents, rels)
        if self._fh is None:
            return
        with self._lock:
            self._fh.write(json.dumps({"k": key, "e": ents, "r": rels}, ensure_ascii=False) + "\n")
            self._fh.flush()

    def tick(self) -> None:
        """청크 1건 완료를 세고, _PROGRESS_EVERY 마다 진행률·ETA 를 찍는다."""
        with self._lock:
            self._done += 1
            done, total = self._done, self._total
        if done % _PROGRESS_EVERY == 0 and total:
            elapsed = time.time() - self._t0
            eta = elapsed / done * (total - done) if done else 0
            _log(f"{done}/{total} ({done / total:.0%}) · 경과 {elapsed / 60:.0f}분 · 남은 ~{eta / 60:.0f}분")

    def close(self) -> None:
        """append 핸들을 닫는다(다음 open 까지 캐시는 메모리에만 남음)."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


_CACHE = _ExtractCache()

_TYPES = "직원/부서/프로젝트/규정/양식/장비/회의/법령/직급/개념"

PROMPT = (
    "아래 사내 문서에서 핵심 엔티티와 그 사이의 관계를 추출하라. "
    "반드시 아래 형식으로만 출력하고, 다른 설명 문장은 쓰지 마라.\n\n"
    "[엔티티]\n"
    f"이름 | 유형 | 한 줄 설명\n(유형은 다음 중 하나: {_TYPES})\n\n"
    "[관계]\n"
    "출발엔티티 | 관계 | 도착엔티티\n\n"
    "문서:\n{text}\n"
)

# 영어/범용 도메인용(외부벤치 HotpotQA 등). 섹션 마커는 파서 호환 위해 한국어 유지,
# 지시문·유형만 영어/범용으로 — 추출 알고리즘·모델·파서는 동일(§동일조건).
PROMPT_EN = (
    "Extract the key entities and the relations between them from the document below. "
    "Output ONLY in the exact format below, with no other explanatory text.\n\n"
    "[엔티티]\n"
    "name | type | one-line description\n"
    "(type is one of: person/place/organization/work/event/group/date/concept)\n\n"
    "[관계]\n"
    "source_entity | relation | target_entity\n\n"
    "문서:\n{text}\n"
)


_DOMAIN_TYPES = {"직원", "부서", "프로젝트", "규정", "양식", "장비", "회의", "법령", "직급", "개념", "회사"}
# 영어 폴백/무라벨 엔티티를 이름 접미사로 재분류 (긴 접미사 우선)
_TYPE_SUFFIX = [
    ("인수인계서", "양식"), ("신청서", "양식"), ("보고서", "양식"), ("청구서", "양식"),
    ("계획서", "양식"), ("명세서", "양식"), ("현황", "양식"), ("통계", "양식"),
    ("규정", "규정"), ("지침", "규정"), ("법률", "법령"), ("법", "법령"),
]
# 관계어미가 엔티티 타입으로 샌 경우 → 올바른 타입
_TYPE_REMAP = {"소속": "직원", "배치부서": "부서", "부서장": "직급"}
# 관계 라벨 표면형 통합(어미 정리로 안 잡히는 특수형만)
_REL_REMAP = {
    "따른다": "따름", "따라 환산": "따름", "따라 지급": "지급", "따라 제출": "제출",
    "따라 정산한다": "정산", "따라 정산": "정산", "준용": "따름", "준수": "따름",
    "근거 제공": "근거", "기반으로 함": "근거", "기초 자료": "근거", "제출한다": "제출",
}


def normalize_name(name: str) -> str:
    """엔티티 이름 정규화(L2 병합용): 인용괄호·따옴표 제거 → 「복지 규정」=복지 규정.
    같은 이름이 되면 LlamaIndex가 id(=name) 기준으로 자동 병합한다.

    입력: name — 추출된 원본 엔티티 이름
    출력: 앞뒤 괄호·따옴표를 벗기고 공백을 정리한 이름
    """
    name = (name or "").strip()
    name = re.sub(r"^[「『“\"'\[\(《]+|[」』”\"'\]\)》]+$", "", name).strip()
    return name


def normalize_type(name: str, typ: str) -> str:
    """엔티티 타입 정규화(결정론적). 도메인 타입은 유지, 오분류는 규칙으로 교정.

    입력: name — 정규화된 엔티티 이름 / typ — LLM이 붙인 유형(빈 문자열 가능)
    출력: 도메인 타입 문자열. 순서는 ① 관계어 오분류 교정(_TYPE_REMAP: 소속→직원)
      ② 도메인 타입이면 그대로 ③ 영어 'entity'·무라벨은 이름 접미사로 재분류
      (_TYPE_SUFFIX: …신청서→양식) ④ 전부 실패하면 "개념"
    """
    typ = (typ or "").strip()
    if typ in _TYPE_REMAP:
        return _TYPE_REMAP[typ]
    if typ in _DOMAIN_TYPES:
        return typ
    # 영어 'entity'/무라벨/기타 → 이름 접미사로 재분류, 없으면 개념
    for suf, t in _TYPE_SUFFIX:
        if name.endswith(suf):
            return t
    return "개념"


def normalize_rel(label: str) -> str:
    """관계 라벨 정규화: 표면형 통합 + 흔한 어미(함/됨/한다/받음) 제거.

    입력: label — 추출된 관계 라벨(예: "따른다", "제출한다")
    출력: 통합된 라벨. _REL_REMAP 특수형이 우선이고, 그 다음 어미를 떼되
      남는 어근이 2글자 미만이면 그대로 둔다(포함→포 방지).
    """
    label = (label or "").strip()
    if label in _REL_REMAP:
        return _REL_REMAP[label]
    # 어미 제거는 남는 어근이 2글자 이상일 때만(예: 포함→포 방지, 포함함→포함 허용).
    for suf in ("한다", "받음", "됨", "함"):
        if label.endswith(suf) and len(label) - len(suf) >= 2:
            return label[: -len(suf)]
    return label


_ROUTER_PROMPT = (
    "질문에 답하려면 '넓은 종합(breadth)'이 필요한지, '특정 사실(pinpoint)'이면 되는지 판단하라.\n"
    "- breadth: 여러 문서·여러 사실·여러 대상을 연결·종합해야 답하는 질문. "
    "다중 홉(A를 알려면 B·C를 거침), 전역·개괄·종합형이 여기 속한다. "
    "'전사/전체/각각/여러/주요/전반적으로/어떻게 이어지나' 신호가 있으면 breadth.\n"
    "  예: '세 사업부는 각각 무엇을 하나', '전사적으로 예산은 어떻게 관리되나', "
    "'연차를 쓰려면 어떤 양식을 작성해 누구 승인을 받나'\n"
    "- pinpoint: 특정 문서 하나의 사실·수치·규정·관계를 콕 집는 질문.\n"
    "  예: '연차는 며칠인가', 'X의 법적 근거는', '휴가신청서 번호는'\n"
    "반드시 한 단어로만 답하라: breadth 또는 pinpoint\n\n질문: {q}\n답:"
)


def parse_extraction(resp: str):
    """E2B 산문 출력 → (entities, relations). entities=(이름,유형,설명), relations=(출발,관계,도착).

    입력: resp — PROMPT 형식(`[엔티티]`/`[관계]` 섹션 + `|` 구분)의 LLM 원문 응답
    출력: (ents, rels) 튜플. 줄머리 불릿을 벗기고 섹션 마커로 상태를 바꾸며,
      엔티티는 2칸 이상(설명 없으면 빈 문자열)·관계는 정확히 3칸이고 전부 비지 않은
      줄만 채택한다. `|` 없는 줄은 무시 → 잡설이 섞여도 안전.
    """
    ents, rels = [], []
    section = None
    for raw in resp.splitlines():
        s = raw.strip().lstrip("*-•·>").strip()
        if not s:
            continue
        low = s.replace(" ", "")
        if "[엔티티]" in low or low.startswith("엔티티"):
            section = "e"
            continue
        if "[관계]" in low or low.startswith("관계"):
            section = "r"
            continue
        if "|" not in s:
            continue
        parts = [p.strip() for p in s.split("|")]
        if section == "e" and len(parts) >= 2:
            name = parts[0]
            typ = parts[1] or "개념"
            desc = parts[2] if len(parts) >= 3 else ""
            if name:
                ents.append((name, typ, desc))
        elif section == "r" and len(parts) == 3 and all(parts):
            rels.append((parts[0], parts[1], parts[2]))
    return ents, rels


class GraphRAGE2B(GraphRAG):
    """E2B 산문 추출 그래프 + E5 재순위 — 이 계열의 기준 구현.

    상위 GraphRAG(평면 트리플)와 달리 엔티티에 유형·설명이 채워지고, 그래프가
    무랭킹으로 뱉는 후보를 e5 유사도로 다시 정렬해 top_k 만 넘긴다.
    """

    name = "graphrag_e2b"

    # 재순위: 그래프에서 후보를 넉넉히(top_k×5, ≥20) 가져와 아래 재랭커로 top_k 재정렬.
    _RETRIEVE_MULT = 5
    _RETRIEVE_MIN = 20

    def _extractor(self) -> Any:
        """산문 추출기를 만든다. cfg.extract_lang 으로 한국어/영어 프롬프트를 고른다.

        출력: _E2BExtractor (추출 LLM은 self.llm — 그래프 품질은 이 모델에 좌우)
          cache_path 는 persist_dir 옆에 두어, 중단 후 재실행 시 남은 청크만 호출한다.
        """
        prompt = PROMPT_EN if getattr(self.cfg, "extract_lang", "ko") == "en" else PROMPT
        return _E2BExtractor(
            llm=self.llm,
            num_workers=1,
            prompt=prompt,
            cache_path=f"{self.persist_dir}_extract.jsonl",
        )

    def _make_engine(self) -> Any:
        """그래프 검색 + E5 재순위 질의 엔진을 만든다.

        출력: LlamaIndex query engine. 후보를 pool(top_k×5, 최소 20)만큼 넉넉히
          받아 _E5Rerank 로 top_k 로 좁힌다.
        """
        import nest_asyncio  # 그래프 검색기의 중첩 async 허용

        nest_asyncio.apply()
        pool = max(self.cfg.top_k * self._RETRIEVE_MULT, self._RETRIEVE_MIN)
        reranker = _E5Rerank(embed_model=self.embed_model, top_n=self.cfg.top_k)
        return self._index.as_query_engine(
            similarity_top_k=pool,
            node_postprocessors=[reranker],
            llm=self._big_llm(),
        )


class GraphRAGE2BL5(GraphRAGE2B):
    """graphrag_e2b + L5 커뮤니티 요약 주입 — 전역(global) 강화.

    질의 시 그래프 검색(재순위) 결과에, 질의와 유사한 '커뮤니티 요약'(압축된 breadth)을
    top-N 붙여 synthesis에 함께 넣는다. 온디바이스 컨텍스트 제약에서 global을 살리는 압축 전략.
    사전: scripts/build_community_summaries.py 로 community_summaries.json 생성.
    """

    name = "graphrag_e2b_l5"
    _TOP_COMMUNITY = 3
    _summaries = None  # [(summary_text, embedding)]

    def __init__(self, cfg: Any, llm: Any, embed_model: Any):
        """백엔드를 초기화하되 인덱스 경로를 graphrag_e2b 로 되돌린다.

        입력: cfg — 설정 / llm — 추출·라우팅용 LLM / embed_model — 재순위·요약 임베딩
        출력: 없음(persist_dir 을 graphrag_e2b 로 고정 → 재추출 없이 그래프 공유)
        """
        super().__init__(cfg, llm, embed_model)
        # L5는 재추출 없이 graphrag_e2b 그래프 + community_summaries.json 재사용.
        self.persist_dir = os.path.join(cfg.storage_dir, "graphrag_e2b")

    def _load_summaries(self) -> Any:
        """community_summaries.json 을 읽어 (요약문, 임베딩) 목록으로 캐시한다.

        출력: [(summary_text, embedding)] 리스트. 파일이 없으면 빈 리스트(요약 주입 비활성).
        """
        if self._summaries is not None:
            return self._summaries
        import json
        path = os.path.join(self.persist_dir, "community_summaries.json")
        if not os.path.exists(path):
            self._summaries = []
            return self._summaries
        data = json.load(open(path))
        texts = [d["summary"] for d in data if d.get("summary")]
        embs = self.embed_model.get_text_embedding_batch(texts) if texts else []
        self._summaries = list(zip(texts, embs))
        return self._summaries

    def _make_engine(self) -> Any:
        """그래프 검색 + 재순위 + 커뮤니티 요약 상시 주입 엔진을 만든다.

        출력: RetrieverQueryEngine. router_llm 을 주지 않으므로 요약은 질의 종류와
          무관하게 항상 top-3 붙는다(전역 강화, 핀포인트는 희석 위험).
        """
        import nest_asyncio

        nest_asyncio.apply()
        from llama_index.core.query_engine import RetrieverQueryEngine

        pool = max(self.cfg.top_k * self._RETRIEVE_MULT, self._RETRIEVE_MIN)
        base = self._index.as_retriever(similarity_top_k=pool)
        reranker = _E5Rerank(embed_model=self.embed_model, top_n=self.cfg.top_k)
        retriever = _GraphCommunityRetriever(
            base=base,
            reranker=reranker,
            embed_model=self.embed_model,
            summaries=self._load_summaries(),
            top_comm=self._TOP_COMMUNITY,
        )
        return RetrieverQueryEngine.from_args(retriever, llm=self._big_llm())


class GraphRAGE2BAdaptive(GraphRAGE2BL5):
    """graphrag_e2b_l5 + 질의 유형 라우팅(적응형 검색).

    질의를 gemma로 global/specific 분류 → 전역형이면 커뮤니티 요약 주입(breadth),
    특정형이면 재순위 top-k만(요약 희석 없음). 핀포인트·전역 양쪽을 균형 있게.
    """

    name = "graphrag_e2b_adaptive"

    def _make_engine(self) -> Any:
        """L5와 같되 라우터 LLM을 달아 요약 주입을 질의 유형에 따라 켠다.

        출력: RetrieverQueryEngine. breadth 로 분류된 질의에만 커뮤니티 요약을 붙이고,
          pinpoint 는 재순위 top_k 만 넘겨 희석을 피한다.
        """
        import nest_asyncio

        nest_asyncio.apply()
        from llama_index.core.query_engine import RetrieverQueryEngine

        pool = max(self.cfg.top_k * self._RETRIEVE_MULT, self._RETRIEVE_MIN)
        base = self._index.as_retriever(similarity_top_k=pool)
        reranker = _E5Rerank(embed_model=self.embed_model, top_n=self.cfg.top_k)
        retriever = _GraphCommunityRetriever(
            base=base,
            reranker=reranker,
            embed_model=self.embed_model,
            summaries=self._load_summaries(),
            top_comm=self._TOP_COMMUNITY,
            router_llm=self.llm,  # ← 질의 유형 라우팅
        )
        return RetrieverQueryEngine.from_args(retriever, llm=self._big_llm())


class GraphRAGE2BHybrid(GraphRAGE2BL5):
    """검색층 하이브리드 — 그래프 검색 + 직접 청크 벡터검색(RRF 융합) + 재순위.

    단일 그래프 검색(query→엔티티→청크, 간접)의 정밀도 약점을,
    직접 벡터검색(query→청크)을 융합해 보완. 의미+키워드 하이브리드가 단일보다
    나았던 것과 같은 원리(그래프+벡터). standard 인덱스를 직접검색으로 재사용.
    """

    name = "graphrag_e2b_hybrid"

    def _make_engine(self) -> Any:
        """그래프 retriever와 standard 벡터 retriever를 RRF로 융합한 엔진을 만든다.

        출력: RetrieverQueryEngine. 두 검색 결과를 reciprocal_rerank 로 합친 뒤
          _E5Rerank 로 top_k 를 고른다. standard 인덱스가 없으면 FileNotFoundError.
        """
        import nest_asyncio

        nest_asyncio.apply()
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import QueryFusionRetriever

        pool = max(self.cfg.top_k * self._RETRIEVE_MULT, self._RETRIEVE_MIN)
        graph_ret = self._index.as_retriever(similarity_top_k=pool)
        # 직접 청크 벡터검색: standard(VectorStoreIndex) 재사용.
        # 주의: standard 는 이 그래프와 반드시 같은 임베딩으로 색인돼 있어야 한다(차원·의미
        # 일치). 다른 임베딩으로 만든 standard 를 쓰면 차원 불일치/무의미 유사도가 된다.
        vdir = os.path.join(self.cfg.storage_dir, "standard")
        if not os.path.isdir(vdir):
            raise FileNotFoundError(
                f"graphrag_e2b_hybrid 는 직접 벡터검색용 'standard' 인덱스가 필요합니다: {vdir} 없음. "
                f"같은 임베딩으로 'ragbench index --method standard' 를 먼저 실행하세요."
            )
        vindex = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vdir), embed_model=self.embed_model
        )
        vec_ret = vindex.as_retriever(similarity_top_k=pool)
        fusion = QueryFusionRetriever(
            [graph_ret, vec_ret],
            similarity_top_k=pool,
            num_queries=1,  # 질의 확장 없이 두 retriever 결과만 RRF 융합
            mode="reciprocal_rerank",
            use_async=False,
            llm=self._big_llm(),
        )
        reranker = _E5Rerank(embed_model=self.embed_model, top_n=self.cfg.top_k)
        return RetrieverQueryEngine.from_args(
            fusion, node_postprocessors=[reranker], llm=self._big_llm()
        )


def _build_community_retriever_cls():
    """그래프 검색(재순위) + 커뮤니티 요약 주입을 합치는 커스텀 retriever.

    출력: GraphCommunityRetriever 클래스(런타임 정의 — 상위 import 순서 의존 회피)
    """
    import math

    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    def _cos(a, b):
        """두 벡터의 코사인 유사도를 계산한다(요약 랭킹용).

        입력: a, b — 같은 차원의 임베딩 벡터
        출력: -1~1 실수(0벡터 대비 분모에 1e-9)
        """
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return s / (na * nb + 1e-9)

    class GraphCommunityRetriever(BaseRetriever):
        """그래프 후보를 재순위한 뒤, 필요하면 커뮤니티 요약 노드를 덧붙이는 retriever."""

        def __init__(self, base, reranker, embed_model, summaries, top_comm=3, router_llm=None):
            """구성요소를 받아 retriever를 초기화한다.

            입력: base — 그래프 retriever / reranker — _E5Rerank /
              embed_model — 질의·요약 임베딩 / summaries — [(요약문, 임베딩)] /
              top_comm — 붙일 요약 개수 / router_llm — None이면 항상 주입(L5),
              주면 breadth 판정 시에만 주입(적응형)
            출력: 없음
            """
            self._base = base
            self._rr = reranker
            self._em = embed_model
            self._summaries = summaries
            self._top = top_comm
            self._router = router_llm  # None=항상 주입(L5) / 있으면 global일 때만(적응형)
            super().__init__()

        def _inject_summaries(self, query_str) -> bool:
            """이 질의에 커뮤니티 요약을 붙일지 판단한다.

            입력: query_str — 사용자 질문
            출력: True면 요약 주입. 요약이 없으면 False, 라우터가 없으면 True(L5),
              있으면 breadth/pinpoint 중 먼저 등장한 라벨을 채택하고 분류 실패 시 True.
            """
            if not self._summaries:
                return False
            if self._router is None:
                return True
            try:
                resp = str(self._router.complete(_ROUTER_PROMPT.format(q=query_str))).strip().lower()
                # multi+global=breadth(주입) / single+relational=pinpoint(재순위만).
                # 장황한 응답 대비 두 라벨의 첫 등장 위치로 판정.
                b, p = resp.find("breadth"), resp.find("pinpoint")
                if b == -1:
                    return False  # breadth 언급 없음 → pinpoint(주입 안 함)
                if p == -1:
                    return True
                return b < p  # 먼저 나온 라벨 채택
            except Exception:
                return True  # 분류 실패 시 안전하게 주입(breadth)

        def _retrieve(self, query_bundle):
            """그래프 검색 → 재순위 → (조건부) 커뮤니티 요약 노드 추가.

            입력: query_bundle — LlamaIndex 질의 번들
            출력: NodeWithScore 목록(요약은 "[커뮤니티 요약] " 접두사로 구분)
            """
            nodes = self._base.retrieve(query_bundle)
            nodes = self._rr.postprocess_nodes(nodes, query_bundle=query_bundle)
            if self._inject_summaries(query_bundle.query_str):
                q = self._em.get_query_embedding(query_bundle.query_str)
                scored = sorted(
                    ((_cos(q, e), t) for t, e in self._summaries),
                    key=lambda x: x[0],
                    reverse=True,
                )
                for sc, txt in scored[: self._top]:
                    nodes.append(
                        NodeWithScore(node=TextNode(text="[커뮤니티 요약] " + txt), score=float(sc))
                    )
            return nodes

    return GraphCommunityRetriever


def _build_reranker_cls():
    """질의-노드 e5 유사도로 재순위하는 node postprocessor(로컬 임베딩 재활용).

    출력: E5Rerank 클래스(런타임 정의 — 상위 import 순서 의존 회피)
    """
    import math

    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import MetadataMode

    class E5Rerank(BaseNodePostprocessor):
        """검색 후보를 질의-본문 e5 코사인 유사도로 다시 정렬해 top_n 만 남긴다.

        그래프 검색은 랭킹 없이 후보를 대량 반환하므로, 이 재순위가 실질적인
        정밀도 장치다. 임베딩 대상 텍스트는 clean() 으로 트리플 접두사를 뗀다.
        """

        embed_model: Any
        top_n: int = 4

        @classmethod
        def class_name(cls) -> str:
            """LlamaIndex 직렬화용 클래스 이름을 돌려준다.

            출력: "E5Rerank"
            """
            return "E5Rerank"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            """후보 노드를 질의 유사도로 재정렬한다.

            입력: nodes — 검색된 NodeWithScore 목록 / query_bundle — 질의(없으면 앞 top_n 만 자름)
            출력: score 를 코사인 유사도로 덮어쓴 상위 top_n 노드. 본문이 빈 노드는
              score=-1.0 으로 밀어낸다.
            """
            if not nodes or query_bundle is None:
                return nodes[: self.top_n]
            q = self.embed_model.get_query_embedding(query_bundle.query_str)

            def cos(a, b):
                """두 벡터의 코사인 유사도를 계산한다.

                입력: a, b — 같은 차원의 임베딩 벡터
                출력: -1~1 실수(0벡터 대비 분모에 1e-9)
                """
                s = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(y * y for y in b))
                return s / (na * nb + 1e-9)

            def clean(t: str) -> str:
                """재순위 임베딩 전에 그래프 노드 앞의 트리플 블록만 떼어낸다.

                입력: t — 노드 본문 텍스트
                출력: 트리플 접두사를 제거한 원문. 이 처리가 없으면 e5-small(512토큰)이
                  앞의 트리플만 임베딩해 정답 청크가 밀려나는 버그가 난다(§10.5).
                """
                # PropertyGraphIndex가 그래프 노드 앞에 붙이는 "Here are some facts…:" +
                # 트리플 블록만 제거하고 원문은 보존한다. e5-small(512토큰)이 앞의 트리플만
                # 임베딩해 정답 청크를 낮게 매기는 버그 방지. 접두사가 없는 노드(직접 벡터검색
                # 청크 등)는 원문의 정당한 '->' 라인을 지우지 않도록 그대로 반환.
                if "Here are some facts" not in t:
                    return t
                out, in_prefix = [], True
                for ln in t.splitlines():
                    if in_prefix and ("Here are some facts" in ln or " -> " in ln or not ln.strip()):
                        continue  # 선두 트리플 블록만 스킵
                    in_prefix = False
                    out.append(ln)
                return "\n".join(out).strip() or t

            texts = [clean(nw.node.get_content(metadata_mode=MetadataMode.NONE) or "") for nw in nodes]
            embs = self.embed_model.get_text_embedding_batch(
                [t if t.strip() else " " for t in texts]
            )
            for nw, t, e in zip(nodes, texts, embs):
                nw.score = cos(q, e) if t.strip() else -1.0
            nodes = sorted(nodes, key=lambda n: (n.score if n.score is not None else -1.0), reverse=True)
            return nodes[: self.top_n]

    return E5Rerank


def _build_extractor_cls():
    """런타임에 TransformComponent 서브클래스를 만든다(상위 import 순서 의존 회피).

    출력: E2BExtractor 클래스
    """
    from llama_index.core.async_utils import run_jobs
    from llama_index.core.graph_stores.types import (
        EntityNode,
        Relation,
        KG_NODES_KEY,
        KG_RELATIONS_KEY,
    )
    from llama_index.core.llms.llm import LLM
    from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

    class E2BExtractor(TransformComponent):
        """산문 형식 추출을 파싱해 청크 메타데이터에 엔티티·관계를 심는 변환 컴포넌트.

        LlamaIndex 의 kg_extractor 자리에 꽂혀, 인덱싱 중 청크마다 LLM을 호출한다.
        JSON 스키마를 강제하지 않으므로 작은 모델도 쓸 수 있고, 대신 파싱과
        타입·관계 정규화를 결정론적 규칙으로 처리한다.
        """

        llm: LLM
        num_workers: int = 1
        prompt: str = PROMPT
        cache_path: str | None = None  # 체크포인트 JSONL. 죽어도 여기서 재개한다.

        @classmethod
        def class_name(cls) -> str:
            """LlamaIndex 직렬화용 클래스 이름을 돌려준다.

            출력: "E2BPathExtractor"
            """
            return "E2BPathExtractor"

        def __call__(self, nodes, show_progress: bool = False, **kwargs):
            """동기 진입점 — 비동기 추출을 이벤트 루프에서 돌린다.

            입력: nodes — 청크 노드 목록 / show_progress — 진행바 표시 여부
            출력: 엔티티·관계가 메타데이터에 채워진 노드 목록
            """
            return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

        async def _aextract(self, node: BaseNode) -> BaseNode:
            """청크 하나에서 엔티티·관계를 뽑아 노드 메타데이터에 기록한다.

            입력: node — 텍스트 청크 노드
            출력: 같은 노드(KG_NODES_KEY/KG_RELATIONS_KEY 채움). 이름·타입·관계를
              정규화하고, 관계에만 등장하는 엔티티도 타입을 추론해 생성한다.
              LLM 호출·파싱이 실패하면 빈 결과로 넘어가 인덱싱을 멈추지 않는다.
            """
            # 청크(≤1024토큰) 전체를 추출에 넣도록 여유 있게(이전 2000자 절단이 긴 청크 손실).
            text = node.get_content(metadata_mode=MetadataMode.LLM)[:4000]
            key = _cache_key(text)
            hit = _CACHE.get(key)
            if hit is not None:  # 체크포인트 재사용 — LLM 호출 없이 즉시 복원
                ents, rels = hit
            else:
                ents, rels = await self._extract_with_retry(text)
                _CACHE.put(key, ents, rels)
            _CACHE.tick()

            kg_nodes = node.metadata.pop(KG_NODES_KEY, [])
            kg_rels = node.metadata.pop(KG_RELATIONS_KEY, [])
            meta = node.metadata.copy()

            name2node = {}
            for name0, typ, desc in ents:
                name = normalize_name(name0)
                if not name:
                    continue
                en = EntityNode(
                    name=name,
                    label=normalize_type(name, typ),
                    properties={**meta, "description": desc},
                )
                name2node[name] = en
                kg_nodes.append(en)
            for src0, rel0, tgt0 in rels:
                src, tgt = normalize_name(src0), normalize_name(tgt0)
                if not src or not tgt:
                    continue
                rel = normalize_rel(rel0)
                sn = name2node.get(src)
                if sn is None:
                    # 관계에만 등장하는 노드: 기본 라벨 'entity' 대신 타입 정규화 적용.
                    sn = EntityNode(name=src, label=normalize_type(src, ""), properties=meta)
                    name2node[src] = sn
                    kg_nodes.append(sn)
                tn = name2node.get(tgt)
                if tn is None:
                    tn = EntityNode(name=tgt, label=normalize_type(tgt, ""), properties=meta)
                    name2node[tgt] = tn
                    kg_nodes.append(tn)
                kg_rels.append(Relation(label=rel, source_id=sn.id, target_id=tn.id, properties=meta))

            node.metadata[KG_NODES_KEY] = kg_nodes
            node.metadata[KG_RELATIONS_KEY] = kg_rels
            return node

        async def _extract_with_retry(self, text: str) -> tuple[list, list]:
            """LLM 추출을 타임아웃·재시도로 감싼다 — 한 청크가 전체를 멈추지 못하게.

            공급자 일시장애(502/503)나 응답 없는 연결에 걸리면 _EXTRACT_TIMEOUT_S 에서
            끊고 지수 백오프로 다시 시도한다. 끝내 실패하면 그 청크만 빈 결과로 넘긴다.

            입력: text — 청크 본문
            출력: (엔티티 목록, 관계 목록). 전부 실패 시 ([], [])
            """
            for attempt in range(_EXTRACT_RETRIES):
                try:
                    resp = await asyncio.wait_for(
                        self.llm.acomplete(self.prompt.format(text=text)),
                        timeout=_EXTRACT_TIMEOUT_S,
                    )
                    return parse_extraction(str(resp))
                except asyncio.TimeoutError:
                    _log(f"타임아웃({_EXTRACT_TIMEOUT_S:.0f}s) 재시도 {attempt + 1}/{_EXTRACT_RETRIES}")
                except Exception as exc:  # 502/503 등 공급자 일시장애
                    _log(f"추출 실패({type(exc).__name__}) 재시도 {attempt + 1}/{_EXTRACT_RETRIES}")
                if attempt < _EXTRACT_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
            return [], []

        async def acall(self, nodes, show_progress: bool = False, **kwargs):
            """모든 청크의 추출을 num_workers 만큼 동시에 실행한다.

            시작 시 체크포인트를 열어 이미 끝낸 청크를 건너뛴다 → 중단 후 재실행하면
            남은 청크만 LLM 을 호출한다(행(hang)·크래시 복구).

            입력: nodes — 청크 노드 목록 / show_progress — 진행바 표시 여부
            출력: 추출 결과가 채워진 노드 목록
            """
            _CACHE.open(self.cache_path, total=len(nodes))
            try:
                jobs = [self._aextract(n) for n in nodes]
                return await run_jobs(
                    jobs, workers=self.num_workers, show_progress=show_progress, desc="E2B 추출"
                )
            finally:
                _CACHE.close()

    return E2BExtractor


_E2BExtractor = _build_extractor_cls()
_E5Rerank = _build_reranker_cls()
_GraphCommunityRetriever = _build_community_retriever_cls()

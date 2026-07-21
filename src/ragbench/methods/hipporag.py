"""HippoRAG2 — 공식 `hipporag` 패키지 어댑터 (OpenIE 지식그래프 + Personalized PageRank).

CLAUDE.md §7 준수: PPR·OpenIE는 **공식 구현 그대로** 사용(추측 재구현 금지). 어댑터는
입출력만 공통 계약(RagBackend)에 맞춘다.

검증 충실(공정성): 검색만 HippoRAG(PPR), **답변 생성은 우리 llm**(다른 기법과 동일 Gemini)
으로 통일 → "검색 메커니즘(PPR)만 변수".
- LLM(OpenIE+HippoRAG 내부): Gemini의 OpenAI 호환 엔드포인트로 연결.
- 임베딩: cfg.hipporag_embedding (기본 facebook/contriever, 경량 로컬). e5 공정비교는
  OpenAI 호환 서버를 띄워 "text-embedding-..." 이름 + embedding_base_url 로 연결.
- recall@k 는 HippoRAG 반환이 원문 문단(파일명 없음)이라 매칭 불가 → judge 로 평가.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from ..core.config import Config
from ..core.interface import QueryResult, RagBackend, RetrievedContext

_GEMINI_OPENAI = "https://generativelanguage.googleapis.com/v1beta/openai/"

_ANSWER_PROMPT = (
    "다음 근거 문단만 사용해 질문에 정확히 답하라. 근거에 없으면 모른다고 하라.\n\n"
    "[근거]\n{ctx}\n\n[질문]\n{q}\n\n[답변]\n"
)

# 공개벤치용 짧은답(EM/F1 비교 가능) — _common._SHORT_QA_TMPL 과 같은 지시를 맞춘다.
_ANSWER_PROMPT_SHORT = (
    "Answer the question using ONLY the context below. Do not use prior knowledge and "
    "do not speculate — every part of the answer must be supported by the context.\n"
    "Give the SHORTEST possible answer span (a name, entity, number, date, or yes/no). "
    "Output only the answer itself — no sentence, no explanation, no punctuation at the end.\n"
    "If the context does not support an answer, output exactly: NO_ANSWER\n\n"
    "[Context]\n{ctx}\n\n[Question]\n{q}\n\n[Answer]\n"
)


def _patch_hipporag_llm() -> None:
    """Gemini 의 OpenAI 호환 엔드포인트는 'seed' 필드를 거부(Unknown name "seed") →
    CacheOpenAI 가 generate_params 에 항상 넣는 seed 를 런타임에 제거(공식 코드 미수정).

    출력: 없음(CacheOpenAI._init_llm_config 를 교체. _ragbench_patched 플래그로 1회만 적용)
    """
    from hipporag.llm.openai_gpt import CacheOpenAI

    if getattr(CacheOpenAI, "_ragbench_patched", False):
        return
    _orig = CacheOpenAI._init_llm_config

    def _init_no_seed(self):
        """원래 설정 초기화를 돌린 뒤 generate_params 에서 seed 만 뺀다.

        출력: 없음(self.llm_config 를 제자리 수정)
        """
        _orig(self)
        self.llm_config.generate_params.pop("seed", None)

    CacheOpenAI._init_llm_config = _init_no_seed
    CacheOpenAI._ragbench_patched = True


def _patch_hipporag_filter() -> None:
    """recognition memory 트리플 필터 파서를 관대하게 — Gemini 등 출력 JSON이 잘려도
    완전한 [주어,관계,목적어] 트리플만 정규식으로 복구한다(재인덱싱 없이 파싱실패 흡수).
    공식 parse_filter 는 json/ast 실패 시 빈 리스트를 반환해 그 질의의 필터가 무력화됨.

    출력: 없음(DSPyFilter.parse_filter 를 교체. _ragbench_repair 플래그로 1회만 적용)
    """
    import json as _json
    import re as _re
    from hipporag.rerank import DSPyFilter

    if getattr(DSPyFilter, "_ragbench_repair", False):
        return
    _field = _re.compile(r"\[\[ ## (\w+) ## \]\]")
    _triple = _re.compile(
        r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]'
    )

    def _parse_repaired(self, response):
        """필터 LLM 응답에서 트리플 목록을 뽑는다(잘린 출력도 복구).

        입력: response — DSPy 형식(`[[ ## 필드 ## ]]` 섹션)의 원문 응답
        출력: [[주어, 관계, 목적어], …]. fact_after_filter 섹션을 우선 보고,
          JSON 파싱이 되면 그 결과를, 안 되면 정규식으로 완전한 트리플만 건진다.
        """
        # fact_after_filter 섹션 추출
        sections, cur = {}, None
        for line in (response or "").splitlines():
            m = _field.match(line.strip())
            if m:
                cur = m.group(1); sections[cur] = []
            elif cur is not None:
                sections[cur].append(line)
        value = "\n".join(sections.get("fact_after_filter", [])).strip() or (response or "")
        # 1) 정상 파싱(json)
        try:
            obj = _json.loads(value)
            facts = obj.get("fact") if isinstance(obj, dict) else obj
            out = [list(t) for t in facts if isinstance(t, (list, tuple)) and len(t) == 3]
            if out:
                return out
        except Exception:
            pass
        # 2) 잘린 출력 → 완전한 트리플만 복구
        return [[a, b, c] for a, b, c in _triple.findall(value)]

    DSPyFilter.parse_filter = _parse_repaired
    DSPyFilter._ragbench_repair = True


def _patch_nim_embedding(api_key: str, model: str = "nvidia/nv-embed-v1") -> None:
    """NIM 임베딩(nv-embed 등)을 HippoRAG OpenAI 임베딩 경로에 연결.
    ① 임베딩 클라이언트에 별도 NVIDIA 키 주입(LLM은 Gemini 키 유지) ② nv-embed 필수
    파라미터 input_type=passage 추가 ③ API 모델명을 실제 NIM 모델로 고정(라우팅용 이름과 분리).

    입력: api_key — NVIDIA API 키 / model — NIM 임베딩 모델 ID
    출력: 없음(OpenAIEmbeddingModel 의 __init__·encode 를 교체. _ragbench_nim 플래그로 1회만 적용)
    """
    import numpy as _np
    from openai import OpenAI as _OpenAI
    from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel

    if getattr(OpenAIEmbeddingModel, "_ragbench_nim", False):
        return
    _orig_init = OpenAIEmbeddingModel.__init__

    def _init(self, global_config=None, embedding_model_name=None):
        """원래 초기화 후 임베딩 클라이언트를 NIM 키·엔드포인트로 갈아끼운다.

        입력: global_config — HippoRAG 전역 설정 / embedding_model_name — 라우팅용 이름
        출력: 없음(self.client 와 self._nim_model 설정)
        """
        _orig_init(self, global_config=global_config, embedding_model_name=embedding_model_name)
        self.client = _OpenAI(base_url=global_config.embedding_base_url, api_key=api_key)
        self._nim_model = model

    def _encode(self, texts):
        """텍스트 배치를 NIM 임베딩 API로 벡터화한다.

        입력: texts — 문자열 목록(개행은 공백으로, 빈 문자열은 " " 로 치환)
        출력: numpy 배열 (len(texts) × 임베딩 차원)
        """
        texts = [t.replace("\n", " ") or " " for t in texts]
        resp = self.client.embeddings.create(
            input=texts, model=self._nim_model, extra_body={"input_type": "passage"}
        )
        return _np.array([v.embedding for v in resp.data])

    OpenAIEmbeddingModel.__init__ = _init
    OpenAIEmbeddingModel.encode = _encode
    OpenAIEmbeddingModel._ragbench_nim = True


class HippoRAGBackend(RagBackend):
    """공식 hipporag 패키지를 공통 RagBackend 계약에 맞추는 얇은 어댑터.

    검색(OpenIE 그래프 + Personalized PageRank)은 공식 구현에 그대로 맡기고,
    답변 생성만 우리 llm 으로 처리한다 → 다른 기법과 생성 조건이 같아
    "검색 메커니즘만 변수"인 공정 비교가 된다. embed_model 인자는 계약을 맞추기 위해
    받지만 쓰지 않는다(임베딩은 HippoRAG 내부 설정으로 지정).
    """

    name = "hipporag"

    def __init__(self, cfg: Config, llm: Any, embed_model: Any):
        """설정·생성 LLM·저장 경로를 잡아둔다(HippoRAG 인스턴스는 지연 생성).

        입력: cfg — 설정 / llm — 답변 생성용 LLM / embed_model — 미사용(계약상 인자)
        출력: 없음
        """
        self.cfg = cfg
        self.llm = llm
        self.save_dir = os.path.join(cfg.storage_dir, self.name)
        self._hr = None

    def _hipporag(self) -> Any:
        """HippoRAG 인스턴스를 몽키패치·키·엔드포인트 배선까지 마쳐 만들고 캐시한다.

        출력: HippoRAG 객체. 내부 LLM은 기본 Gemini OpenAI호환(config 로 NIM 등 override),
          임베딩은 cfg.hipporag_embedding(기본 contriever)이며 embedding_base_url 을 주면
          e5 등 OpenAI호환 서버를 쓴다. hipporag 는 여기서 지연 import(선택 의존성).
        """
        if self._hr is not None:
            return self._hr
        from hipporag import HippoRAG

        _patch_hipporag_llm()
        _patch_hipporag_filter()  # 트리플필터 파싱 관대화(잘린 JSON 복구, 재인덱싱 불필요)
        # HippoRAG 내부 LLM(OpenIE+트리플필터) — 기본 Gemini OpenAI호환, override로 NIM Llama 등.
        llm_base = getattr(self.cfg, "hipporag_llm_base_url", None) or _GEMINI_OPENAI
        llm_model = getattr(self.cfg, "hipporag_llm_model", None) or self.cfg.llm.model
        key_env = getattr(self.cfg, "hipporag_llm_key_env", None) or "GEMINI_API_KEY"
        # HippoRAG 의 OpenAI 클라이언트는 OPENAI_API_KEY 를 읽는다 → 지정 키로 채움.
        key = os.environ.get(key_env) or os.environ.get("GOOGLE_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key
        emb = getattr(self.cfg, "hipporag_embedding", None) or "facebook/contriever"
        emb_url = getattr(self.cfg, "hipporag_embedding_base_url", None)
        if emb_url and "nvidia.com" in emb_url:  # NIM 임베딩: 별도 NVIDIA 키 + input_type 패치
            nv_key = os.environ.get("NVIDIA_API_KEY")
            if nv_key:
                _patch_nim_embedding(nv_key, model=os.environ.get("NIM_EMBED_MODEL", "nvidia/nv-embed-v1"))
        kwargs = dict(
            save_dir=self.save_dir,
            llm_model_name=llm_model,
            llm_base_url=llm_base,
            embedding_model_name=emb,
        )
        if emb_url:  # e5 등 OpenAI호환 서버로 임베딩(공정비교) — 이름에 "text-embedding" 필요
            kwargs["embedding_base_url"] = emb_url
        self._hr = HippoRAG(**kwargs)
        return self._hr

    def index(self, documents: Sequence[Any]) -> None:
        """문서 본문을 넘겨 HippoRAG의 OpenIE 지식그래프를 구축한다.

        입력: documents — Document 객체 또는 문자열 목록
        출력: 없음(save_dir 아래에 그래프·임베딩 영속화)
        """
        docs = [d.get_content() if hasattr(d, "get_content") else str(d) for d in documents]
        self._hipporag().index(docs=docs)

    def query(self, question: str) -> QueryResult:
        """PPR로 근거 문단을 검색하고, 우리 LLM으로 답을 생성한다.

        입력: question — 사용자 질문
        출력: QueryResult(answer, contexts, metadata). contexts 는 원문 문단이라
          파일명이 없어 recall@k 는 산출 불가 → judge·kw_recall 로 평가한다.
        """
        sols = self._hipporag().retrieve(
            queries=[question], num_to_retrieve=self.cfg.top_k
        )
        sol = sols[0] if isinstance(sols, (list, tuple)) else sols
        docs_raw = getattr(sol, "docs", None)
        scores_raw = getattr(sol, "doc_scores", None)
        docs = list(docs_raw) if docs_raw is not None else []
        scores = list(scores_raw) if scores_raw is not None else []
        contexts = [
            RetrievedContext(text=d, score=(scores[i] if i < len(scores) else None))
            for i, d in enumerate(docs)
        ]
        ctx = "\n\n".join(f"- {d}" for d in docs) or "(근거 없음)"
        tmpl = (
            _ANSWER_PROMPT_SHORT
            if getattr(self.cfg, "answer_style", "default") == "short"
            else _ANSWER_PROMPT
        )
        answer = str(self.llm.complete(tmpl.format(ctx=ctx[:8000], q=question)))
        return QueryResult(
            answer=answer,
            contexts=contexts,
            metadata={"method": self.name, "top_k": self.cfg.top_k, "retriever": "PPR"},
        )

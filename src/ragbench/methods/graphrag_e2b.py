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
import os
import re
from typing import Any, Sequence

from .graphrag import GraphRAG

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
    같은 이름이 되면 LlamaIndex가 id(=name) 기준으로 자동 병합한다."""
    name = (name or "").strip()
    name = re.sub(r"^[「『“\"'\[\(《]+|[」』”\"'\]\)》]+$", "", name).strip()
    return name


def normalize_type(name: str, typ: str) -> str:
    """엔티티 타입 정규화(결정론적). 도메인 타입은 유지, 오분류는 규칙으로 교정."""
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
    """관계 라벨 정규화: 표면형 통합 + 흔한 어미(함/됨/한다/받음) 제거."""
    label = (label or "").strip()
    if label in _REL_REMAP:
        return _REL_REMAP[label]
    # 어미 제거는 남는 어근이 2글자 이상일 때만(예: 포함→포 방지, 포함함→포함 허용).
    for suf in ("한다", "받음", "됨", "함"):
        if label.endswith(suf) and len(label) - len(suf) >= 2:
            return label[: -len(suf)]
    return label


_ROUTER_PROMPT = (
    "다음 질문을 두 유형 중 하나로 분류하라.\n"
    "- global: 회사 전체·여러 부서·여러 문서를 아우르는 개괄·요약·종합 질문. "
    "'전사/전체/각각/여러/주요/전반적으로/어떻게 이어지나' 같은 신호가 있으면 global.\n"
    "  예: '세 사업부는 각각 어떤 프로젝트를 하나', '전사적으로 예산은 어떻게 관리되나', '주요 활동은'\n"
    "- specific: 특정 규정·수치·인물·양식 하나를 콕 집어 묻는 질문.\n"
    "  예: '연차는 며칠인가', '휴가신청서 번호는', 'X 규정의 마감일은'\n"
    "반드시 한 단어로만 답하라: global 또는 specific\n\n질문: {q}\n답:"
)


def parse_extraction(resp: str):
    """E2B 산문 출력 → (entities, relations). entities=(이름,유형,설명), relations=(출발,관계,도착)."""
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
    name = "graphrag_e2b"

    # 재순위: 그래프에서 후보를 넉넉히(top_k×5, ≥20) 가져와 아래 재랭커로 top_k 재정렬.
    _RETRIEVE_MULT = 5
    _RETRIEVE_MIN = 20

    def _extractor(self) -> Any:
        return _E2BExtractor(llm=self.llm, num_workers=1)

    def _make_engine(self) -> Any:
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
        super().__init__(cfg, llm, embed_model)
        # L5는 재추출 없이 graphrag_e2b 그래프 + community_summaries.json 재사용.
        self.persist_dir = os.path.join(cfg.storage_dir, "graphrag_e2b")

    def _load_summaries(self) -> Any:
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
    """그래프 검색(재순위) + 커뮤니티 요약 주입을 합치는 커스텀 retriever."""
    import math

    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    def _cos(a, b):
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return s / (na * nb + 1e-9)

    class GraphCommunityRetriever(BaseRetriever):
        def __init__(self, base, reranker, embed_model, summaries, top_comm=3, router_llm=None):
            self._base = base
            self._rr = reranker
            self._em = embed_model
            self._summaries = summaries
            self._top = top_comm
            self._router = router_llm  # None=항상 주입(L5) / 있으면 global일 때만(적응형)
            super().__init__()

        def _inject_summaries(self, query_str) -> bool:
            if not self._summaries:
                return False
            if self._router is None:
                return True
            try:
                resp = str(self._router.complete(_ROUTER_PROMPT.format(q=query_str))).strip().lower()
                # 장황한 응답("specific (not global)" 등) 대비: 두 라벨의 첫 등장 위치로 판정.
                g, s = resp.find("global"), resp.find("specific")
                if g == -1:
                    return False  # global 언급 없음 → specific(주입 안 함)
                if s == -1:
                    return True
                return g < s  # 먼저 나온 라벨 채택
            except Exception:
                return True  # 분류 실패 시 안전하게 주입(전역 breadth)

        def _retrieve(self, query_bundle):
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
    """질의-노드 e5 유사도로 재순위하는 node postprocessor(로컬 임베딩 재활용)."""
    import math

    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import MetadataMode

    class E5Rerank(BaseNodePostprocessor):
        embed_model: Any
        top_n: int = 4

        @classmethod
        def class_name(cls) -> str:
            return "E5Rerank"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            if not nodes or query_bundle is None:
                return nodes[: self.top_n]
            q = self.embed_model.get_query_embedding(query_bundle.query_str)

            def cos(a, b):
                s = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(y * y for y in b))
                return s / (na * nb + 1e-9)

            def clean(t: str) -> str:
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
    """런타임에 TransformComponent 서브클래스를 만든다(상위 import 순서 의존 회피)."""
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
        llm: LLM
        num_workers: int = 1

        @classmethod
        def class_name(cls) -> str:
            return "E2BPathExtractor"

        def __call__(self, nodes, show_progress: bool = False, **kwargs):
            return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

        async def _aextract(self, node: BaseNode) -> BaseNode:
            # 청크(≤1024토큰) 전체를 추출에 넣도록 여유 있게(이전 2000자 절단이 긴 청크 손실).
            text = node.get_content(metadata_mode=MetadataMode.LLM)[:4000]
            try:
                resp = await self.llm.acomplete(PROMPT.format(text=text))
                ents, rels = parse_extraction(str(resp))
            except Exception:
                ents, rels = [], []

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

        async def acall(self, nodes, show_progress: bool = False, **kwargs):
            jobs = [self._aextract(n) for n in nodes]
            return await run_jobs(jobs, workers=self.num_workers, show_progress=show_progress, desc="E2B 추출")

    return E2BExtractor


_E2BExtractor = _build_extractor_cls()
_E5Rerank = _build_reranker_cls()
_GraphCommunityRetriever = _build_community_retriever_cls()

"""기법 레지스트리 — 이름 → RagBackend 구현 매핑.

새 RAG 기법은 methods/ 에 어댑터를 만들고 여기 한 줄만 추가하면 된다 (CLAUDE.md §7).
"""
from __future__ import annotations

from typing import Any

from .core.config import Config
from .core.interface import RagBackend
from .methods.bm25 import BM25RAG
from .methods.graphrag import GraphRAG, GraphRAGDynamic, GraphRAGSchema
from .methods.graphrag_e2b import (
    GraphRAGE2B,
    GraphRAGE2BAdaptive,
    GraphRAGE2BHybrid,
    GraphRAGE2BL5,
)
from .methods.hipporag import HippoRAGBackend
from .methods.hybrid import HybridRAG
from .methods.standard import StandardRAG

METHODS: dict[str, type[RagBackend]] = {
    "standard": StandardRAG,          # 의미(벡터) 검색
    "bm25": BM25RAG,                  # 키워드(렉시컬) 검색
    "hybrid": HybridRAG,              # 벡터 + BM25 (RRF)
    "graphrag": GraphRAG,             # 그래프 — 평면 트리플(기준선)
    "graphrag_schema": GraphRAGSchema,    # 그래프 — 스키마 강제 + 속성
    "graphrag_dynamic": GraphRAGDynamic,  # 그래프 — 동적 온톨로지 + 속성
    "graphrag_e2b": GraphRAGE2B,          # 그래프 — 로컬 산문 추출 + description(파싱)
    "graphrag_e2b_l5": GraphRAGE2BL5,     # 위 + L5 커뮤니티 요약 주입(전역 강화)
    "graphrag_e2b_adaptive": GraphRAGE2BAdaptive,  # 위 + 질의 유형 라우팅(적응형)
    "graphrag_e2b_hybrid": GraphRAGE2BHybrid,      # 검색층 하이브리드(그래프+직접벡터 RRF)
    "hipporag": HippoRAGBackend,          # OpenIE KG + Personalized PageRank (공식 hipporag)
    # "lightrag": LightRAG,
    # ... 부록 A 참고
}


def build_backend(name: str, cfg: Config, llm: Any, embed_model: Any) -> RagBackend:
    if name not in METHODS:
        raise ValueError(f"알 수 없는 기법: {name!r}. 사용 가능: {sorted(METHODS)}")
    return METHODS[name](cfg, llm, embed_model)

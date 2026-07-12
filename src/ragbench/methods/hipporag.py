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


def _patch_hipporag_llm() -> None:
    """Gemini 의 OpenAI 호환 엔드포인트는 'seed' 필드를 거부(Unknown name "seed") →
    CacheOpenAI 가 generate_params 에 항상 넣는 seed 를 런타임에 제거(공식 코드 미수정)."""
    from hipporag.llm.openai_gpt import CacheOpenAI

    if getattr(CacheOpenAI, "_ragbench_patched", False):
        return
    _orig = CacheOpenAI._init_llm_config

    def _init_no_seed(self):
        _orig(self)
        self.llm_config.generate_params.pop("seed", None)

    CacheOpenAI._init_llm_config = _init_no_seed
    CacheOpenAI._ragbench_patched = True


class HippoRAGBackend(RagBackend):
    name = "hipporag"

    def __init__(self, cfg: Config, llm: Any, embed_model: Any):
        self.cfg = cfg
        self.llm = llm
        self.save_dir = os.path.join(cfg.storage_dir, self.name)
        self._hr = None

    def _hipporag(self) -> Any:
        if self._hr is not None:
            return self._hr
        from hipporag import HippoRAG

        _patch_hipporag_llm()
        # HippoRAG 의 OpenAI 클라이언트는 OPENAI_API_KEY 를 읽는다 → Gemini 키로 채움.
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = key
        emb = getattr(self.cfg, "hipporag_embedding", None) or "facebook/contriever"
        self._hr = HippoRAG(
            save_dir=self.save_dir,
            llm_model_name=self.cfg.llm.model,
            llm_base_url=_GEMINI_OPENAI,
            embedding_model_name=emb,
        )
        return self._hr

    def index(self, documents: Sequence[Any]) -> None:
        docs = [d.get_content() if hasattr(d, "get_content") else str(d) for d in documents]
        self._hipporag().index(docs=docs)

    def query(self, question: str) -> QueryResult:
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
        answer = str(self.llm.complete(_ANSWER_PROMPT.format(ctx=ctx[:8000], q=question)))
        return QueryResult(
            answer=answer,
            contexts=contexts,
            metadata={"method": self.name, "top_k": self.cfg.top_k, "retriever": "PPR"},
        )

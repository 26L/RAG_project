"""BM25 RAG — 키워드(렉시컬) 검색 → top-k 청크 → LLM 생성."""
from __future__ import annotations

from typing import Any

from ._common import LlamaIndexBackend, ko_tokenize


class BM25RAG(LlamaIndexBackend):
    """BM25 키워드(렉시컬) 점수만으로 top-k 청크를 뽑는 백엔드."""

    name = "bm25"

    def _make_engine(self) -> Any:
        """docstore 노드로 BM25 검색기를 만들어 질의 엔진에 연결한다.

        출력: 한국어 토크나이저(ko_tokenize)를 쓰는 BM25Retriever 기반 RetrieverQueryEngine
        """
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.retrievers.bm25 import BM25Retriever

        retriever = BM25Retriever.from_defaults(
            nodes=self._nodes(),
            similarity_top_k=self.cfg.top_k,
            tokenizer=ko_tokenize,
        )
        return RetrieverQueryEngine.from_args(retriever, llm=self.llm)

"""Standard RAG — 벡터(의미) 검색 → top-k 청크 → LLM 생성 (베이스라인)."""
from __future__ import annotations

from typing import Any

from ._common import LlamaIndexBackend


class StandardRAG(LlamaIndexBackend):
    """벡터(의미) 유사도만으로 top-k 청크를 뽑는 베이스라인 백엔드."""

    name = "standard"

    def _make_engine(self) -> Any:
        """인덱스의 기본 벡터 검색 질의 엔진을 만든다.

        출력: similarity_top_k=cfg.top_k 로 설정된 LlamaIndex 질의 엔진
        """
        return self._index.as_query_engine(similarity_top_k=self.cfg.top_k)

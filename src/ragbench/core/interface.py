"""모든 RAG 기법이 따르는 공통 계약.

평가 하니스는 구체 기법을 모른 채 이 인터페이스로만 상호작용한다.
새 기법은 RagBackend 를 구현하는 어댑터로 추가한다 (CLAUDE.md §4).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class RetrievedContext:
    """답변 근거로 쓰인 단일 청크/노드.

    text — 근거 본문 / source — 출처 식별자(파일명 등) / score — 검색 점수 / metadata — 기법별 부가정보.
    """
    text: str
    source: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """기법 무관 공통 응답.

    answer — 생성된 답변 / contexts — 근거 청크 목록 / metadata — 기법별 부가정보.

    metadata 에는 그래프 경로·연결 근거·타이밍 등 기법별 부가정보를 담아,
    평가 하니스가 '연결성 품질'까지 측정할 수 있게 한다 (CLAUDE.md §4.1).
    """
    answer: str
    contexts: list[RetrievedContext] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RagBackend(ABC):
    """RAG 기법 공통 인터페이스.

    name — 기법 식별자. 구현체는 index()/query() 두 메서드만 채우면 하니스가 그대로 쓴다.
    """

    name: str = "base"

    @abstractmethod
    def index(self, documents: Sequence[Any]) -> None:
        """문서를 인덱싱하고 영속화한다.

        입력: documents — 로드된 문서 시퀀스
        출력: 없음(인덱스를 storage_dir 아래에 저장)
        """

    @abstractmethod
    def query(self, question: str) -> QueryResult:
        """질문에 대해 근거 기반 답변을 생성한다.

        입력: question — 질문 텍스트
        출력: QueryResult(답변 + 근거 청크 + 기법별 metadata)
        """

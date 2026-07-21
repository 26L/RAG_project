"""평가셋 로딩. 기법과 무관한 공통 Q/A 형식 (CLAUDE.md §6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EvalItem:
    """평가셋 한 문항. YAML 한 항목이 이 필드들로 그대로 매핑된다.

    필드: question — 질의 / relevant_sources — 정답 근거 문서(파일명), 검색 지표 채점 기준
          answer_keywords — 답변에 있어야 할 키워드(keyword_recall 용)
          reference_answer — LLM-as-judge 채점용 참조 정답
          type — 질문 유형(single/multi/relational/global), 유형별 집계 키
    """

    question: str
    # 정답 근거 문서(파일명) — 검색 지표(recall@k 등) 채점 기준.
    relevant_sources: list[str] = field(default_factory=list)
    # 답변 품질의 거친 프록시 — 답변에 이 키워드들이 들어있는 비율. 없으면 채점 생략.
    answer_keywords: list[str] = field(default_factory=list)
    # 참고용 정답(LLM-as-judge 채점에 사용).
    reference_answer: str | None = None
    # 질문 유형 — single(단일 문서) / multi(다중 홉) / relational(관계·인과). 유형별 지표 분해용.
    type: str = "single"


def load_eval_set(path: str) -> list[EvalItem]:
    """평가셋 YAML(항목 리스트)을 읽어 EvalItem 리스트로 만든다.

    입력: path — YAML 경로. 각 항목 키는 EvalItem 필드명과 동일
          (question 필수, relevant_sources·answer_keywords·reference_answer·type 은 선택)
    출력: EvalItem 리스트 (파일이 비면 빈 리스트)
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [EvalItem(**item) for item in raw]

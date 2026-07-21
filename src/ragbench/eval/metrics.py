"""검색/답변 지표. 기법 무관 — QueryResult 의 contexts(출처)와 answer 만 사용한다."""
from __future__ import annotations

from typing import Sequence


def dedupe_preserve(sources: Sequence[str | None]) -> list[str]:
    """순서 유지하며 출처 중복 제거(None 제외). 같은 문서의 여러 청크는 1개로 본다.

    입력: sources — 검색 순위 순서의 출처 파일명 시퀀스(None 섞여도 됨)
    출력: 첫 등장 순서를 유지한 고유 출처 리스트
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in sources:
        if s is None or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def retrieval_metrics(
    retrieved_sources: Sequence[str | None],
    relevant_sources: Sequence[str],
    k: int,
) -> dict[str, float | None]:
    """문서(출처) 단위 검색 지표. relevant 가 비면 None(채점 생략).

    출처를 중복 제거해 상위 k개만 남기고, 정답 근거 파일명 집합과 교집합(found)으로 계산한다.
      - hit: found 가 하나라도 있으면 1.0, 없으면 0.0
      - recall: len(found) / len(relevant) — 정답 근거 중 top-k 안에 든 비율
      - precision: len(found) / len(ranked) — top-k 중 정답 근거인 비율
      - mrr: 정답 근거가 처음 나온 순위의 역수(1/rank), 없으면 0.0

    입력: retrieved_sources — 검색된 출처(순위순) / relevant_sources — 정답 근거 파일명
          k — 상위 몇 개까지 볼지
    출력: {"hit","recall","precision","mrr"} — relevant 가 비면 값이 전부 None
    """
    relevant = set(relevant_sources)
    if not relevant:
        return {"hit": None, "recall": None, "precision": None, "mrr": None}

    ranked = dedupe_preserve(retrieved_sources)[:k]
    found = set(ranked) & relevant

    mrr = 0.0
    for rank, s in enumerate(ranked, start=1):
        if s in relevant:
            mrr = 1.0 / rank
            break

    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(relevant),
        "precision": len(found) / len(ranked) if ranked else 0.0,
        "mrr": mrr,
    }


def keyword_recall(answer: str, keywords: Sequence[str]) -> float | None:
    """답변 품질의 거친 프록시 — 키워드 포함 비율. 키워드 없으면 None.

    대소문자 무시 부분문자열 매칭으로 포함된 키워드 수 / 전체 키워드 수.

    입력: answer — 모델 답변 / keywords — 정답에 들어가야 할 키워드
    출력: 0.0~1.0 포함 비율, 키워드가 없으면 None(채점 생략)
    """
    if not keywords:
        return None
    low = answer.lower()
    present = sum(1 for kw in keywords if kw.lower() in low)
    return present / len(keywords)

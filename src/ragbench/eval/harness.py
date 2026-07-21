"""평가 하니스 — 같은 평가셋을 어떤 기법에든 돌려 지표를 집계한다 (CLAUDE.md §6).

기법을 전혀 모른 채 공통 RagBackend.query() 만 호출한다.
질문 유형(single/multi/relational)별 분해와 선택적 LLM-as-judge 채점을 지원한다.
"""
from __future__ import annotations

import time
from statistics import mean
from typing import Any, Sequence

from ..core.interface import RagBackend
from .dataset import EvalItem
from .judge import judge_answer
from .metrics import dedupe_preserve, keyword_recall, retrieval_metrics

_AGG_KEYS = ("hit", "recall", "precision", "mrr", "keyword_recall", "judge_correct", "latency_s")


def _avg(values: Sequence[Any]) -> float | None:
    """None 을 빼고 평균낸다(채점 생략된 문항이 평균을 깎지 않게).

    입력: values — 숫자 또는 None 이 섞인 값들
    출력: 소수 4자리 반올림 평균, 유효값이 하나도 없으면 None
    """
    nums = [v for v in values if v is not None]
    return round(mean(nums), 4) if nums else None


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """문항별 결과를 지표별 평균으로 집계한다.

    입력: rows — run_eval 이 만든 문항 row 리스트
    출력: _AGG_KEYS(hit·recall·precision·mrr·keyword_recall·judge_correct·latency_s)
          각각의 평균 + 문항 수 "n"
    """
    agg = {key: _avg([r[key] for r in rows]) for key in _AGG_KEYS}
    agg["n"] = len(rows)
    return agg


def run_eval(
    backend: RagBackend,
    items: Sequence[EvalItem],
    k: int,
    judge_llm: Any | None = None,
) -> dict[str, Any]:
    """평가셋 전 문항을 백엔드에 돌리고 전체·유형별 지표를 집계한다.

    문항마다 query() 지연을 재고, 검색 지표(retrieval_metrics)·keyword_recall·
    (judge_llm 이 있으면) LLM-judge 를 채점한다. 유형별 집계는 row 의 type 값
    (single/multi/relational/global)으로 그룹핑해 같은 방식으로 평균낸다.

    입력: backend — 공통 RagBackend(기법을 몰라도 됨) / items — 평가 문항
          k — 검색 지표 상위 k / judge_llm — 채점용 LLM, None 이면 judge 생략
    출력: {"aggregate": 전체 평균, "by_type": 유형별 평균, "per_item": 문항별 상세
          (질문·유형·지연·지표·top-k 출처·답변)}
    """
    per_item: list[dict[str, Any]] = []
    for it in items:
        t0 = time.perf_counter()
        res = backend.query(it.question)
        latency = round(time.perf_counter() - t0, 3)

        retrieved = dedupe_preserve([c.source for c in res.contexts])
        row = {
            "question": it.question,
            "type": it.type,
            "latency_s": latency,
            **retrieval_metrics(retrieved, it.relevant_sources, k),
            "keyword_recall": keyword_recall(res.answer, it.answer_keywords),
            "judge_correct": (
                judge_answer(judge_llm, it.question, it.reference_answer, res.answer)
                if judge_llm is not None
                else None
            ),
            "retrieved": retrieved[:k],
            "answer": res.answer,
        }
        per_item.append(row)

    # 유형별 분해
    types = sorted({r["type"] for r in per_item})
    by_type = {t: _aggregate([r for r in per_item if r["type"] == t]) for t in types}

    return {
        "aggregate": _aggregate(per_item),
        "by_type": by_type,
        "per_item": per_item,
    }

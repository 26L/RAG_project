"""짧은답 프롬프트 전/후 비교 — judge·EM·F1·답변 길이를 한 표로.

입력: results/local_eval/{method}_hotpot.json(전) 와 {method}_hotpot_short.json(후)
      + config/eval_hotpot.yaml(참조 정답)
출력: stdout 비교표. 전(before)은 EM/F1 이 저장돼 있지 않으므로 답변에서 재계산한다.
"""
from __future__ import annotations

import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from ragbench.eval.metrics import exact_match, token_f1  # noqa: E402

METHODS = ["standard", "bm25", "hybrid", "graphrag_e2b", "hipporag"]
RES = "results/local_eval"


def refs() -> dict[str, str]:
    """평가셋에서 질문→참조정답 매핑을 만든다."""
    data = yaml.safe_load(open("config/eval_hotpot.yaml", encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    return {it["question"]: it.get("reference_answer") for it in items}


def score(path: str, ref: dict[str, str]) -> dict | None:
    """결과 파일 하나를 읽어 judge·EM·F1·평균 답변 단어수를 계산한다."""
    if not os.path.exists(path):
        return None
    rows = json.load(open(path, encoding="utf-8"))["report"]["per_item"]
    ems, f1s, judges, lens = [], [], [], []
    abstain = 0
    for r in rows:
        gold = ref.get(r["question"])
        ans = r["answer"]
        if "NO_ANSWER" in ans or "no information" in ans.lower():
            abstain += 1
        em, f1 = exact_match(ans, gold), token_f1(ans, gold)
        if em is not None:
            ems.append(em)
            f1s.append(f1)
        if r.get("judge_correct") is not None:
            judges.append(r["judge_correct"])
        lens.append(len(ans.split()))
    return {
        "n": len(rows),
        "judge": round(mean(judges), 3) if judges else None,
        "em": round(mean(ems), 3) if ems else None,
        "f1": round(mean(f1s), 3) if f1s else None,
        "words": round(mean(lens), 1),
        "abstain": round(abstain / len(rows), 3),
    }


def main() -> None:
    """전/후 결과를 모아 비교표를 출력한다."""
    ref = refs()
    print(
        f"{'method':16s} {'':>6s} {'judge':>7s} {'EM':>7s} {'F1':>7s} "
        f"{'words':>7s} {'abstain':>8s}"
    )
    print("-" * 65)
    for m in METHODS:
        for tag, suffix in (("before", "_hotpot"), ("after", "_hotpot_short")):
            s = score(f"{RES}/{m}{suffix}.json", ref)
            if s is None:
                continue
            print(
                f"{m:16s} {tag:>6s} {str(s['judge']):>7s} {str(s['em']):>7s} "
                f"{str(s['f1']):>7s} {str(s['words']):>7s} {str(s['abstain']):>8s}"
            )
    print("\n* words = 모델 답변 평균 단어수 (HotpotQA 정답 평균 ≈ 2.2)")
    print("* abstain = 근거 부족으로 기권(NO_ANSWER)한 비율 — 오답이 아니라 검색 실패 신호")


if __name__ == "__main__":
    main()

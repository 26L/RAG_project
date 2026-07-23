"""judge 신뢰도 검증 — 사람 라벨과 LLM-judge 의 일치도(정확도·Cohen's κ)를 잰다.

왜 필요한가: judge 는 이 벤치마크의 **주력 지표**(§6.0)인데 사람 대조가 한 번도 없다.
자(尺)가 검증되지 않으면 그 위에 쌓은 모든 결론이 흔들린다. §10.5 에서 약judge 가
결론을 통째로 뒤집은 전례가 있어 더더욱 필요하다.

2단계로 쓴다.
  1) sample — 결과 파일에서 문항을 뽑아 채점용 시트(YAML)를 만든다. judge 판정은
     **숨긴다**(보고 채점하면 앵커링으로 일치도가 부풀려진다).
  2) score — 사람이 human 칸을 채운 시트를 읽어 정확도·κ·불일치 목록을 낸다.

사용:
  .venv/bin/python scripts/judge_validation.py sample --results results/local_eval/standard_hotpot400.json --n 50
  # → judge_labels.yaml 의 human: null 을 1/0 으로 채운 뒤
  .venv/bin/python scripts/judge_validation.py score --sheet judge_labels.yaml
"""
from __future__ import annotations

import argparse
import json
import random

import yaml


def cmd_sample(args: argparse.Namespace) -> None:
    """결과 파일에서 문항을 무작위 추출해 사람 채점용 시트를 쓴다.

    judge 판정은 시트에 넣지 않는다(앵커링 방지). 대신 재현용 시드와 원본 인덱스를 남겨
    score 단계에서 원본 파일의 judge 값과 대조한다.

    입력: args.results — 결과 JSON / args.n — 표본 수 / args.seed / args.out
    출력: 없음(YAML 시트 파일 생성)
    """
    rows = json.load(open(args.results, encoding="utf-8"))["report"]["per_item"]
    idx = list(range(len(rows)))
    random.Random(args.seed).shuffle(idx)
    picked = sorted(idx[: args.n])

    sheet = {
        "source": args.results,
        "seed": args.seed,
        "instruction": (
            "각 항목의 human 을 1(모델 답변이 참조 정답과 핵심 사실이 일치) 또는 0 으로 채운다. "
            "판단이 불가능하면 null 로 두면 집계에서 제외된다. "
            "★ 기권(NO_ANSWER)은 0 으로 채운다 — 검색이 근거를 못 찾은 것이므로 "
            "검색 성능 평가에서는 실패가 맞다. "
            "★ judge 판정은 일부러 넣지 않았다(보고 채점하면 앵커링으로 일치도가 부풀려진다)."
        ),
        "items": [
            {
                "i": i,
                "question": rows[i]["question"],
                "reference": rows[i].get("reference_answer"),
                "answer": rows[i]["answer"],
                "human": None,
            }
            for i in picked
        ],
    }
    # 참조 정답은 결과 JSON 에 없으므로 평가셋에서 보충
    if args.eval_set:
        ref = {
            it["question"]: it.get("reference_answer")
            for it in yaml.safe_load(open(args.eval_set, encoding="utf-8"))
        }
        for it in sheet["items"]:
            it["reference"] = ref.get(it["question"])

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(sheet, f, allow_unicode=True, sort_keys=False)
    print(f"채점 시트 {len(picked)}문항 → {args.out}  (human 칸을 1/0 으로 채우세요)")


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """두 이진 라벨열의 Cohen's κ — 우연 일치를 보정한 일치도.

    입력: a, b — 같은 길이의 0/1 라벨
    출력: κ (1=완전일치, 0=우연 수준). 기대일치가 1이면 1.0 반환
    """
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cmd_score(args: argparse.Namespace) -> None:
    """채워진 시트를 읽어 judge 와 사람 라벨의 정확도·κ·불일치를 출력한다.

    입력: args.sheet — human 이 채워진 YAML
    출력: 없음(일치도 지표와 불일치 문항을 stdout 에 출력)
    """
    sheet = yaml.safe_load(open(args.sheet, encoding="utf-8"))
    rows = json.load(open(sheet["source"], encoding="utf-8"))["report"]["per_item"]

    pairs = [
        (int(it["human"]), int(rows[it["i"]]["judge_correct"]), it)
        for it in sheet["items"]
        if it["human"] is not None and rows[it["i"]].get("judge_correct") is not None
    ]
    if not pairs:
        print("채점된 항목이 없습니다. human 칸을 1/0 으로 채우세요.")
        return

    human = [p[0] for p in pairs]
    judge = [p[1] for p in pairs]
    n = len(pairs)
    agree = sum(1 for h, j in zip(human, judge) if h == j)
    fp = sum(1 for h, j in zip(human, judge) if j == 1 and h == 0)  # judge 관대
    fn = sum(1 for h, j in zip(human, judge) if j == 0 and h == 1)  # judge 엄격

    print(f"n={n} · 사람 정답률={sum(human)/n:.3f} · judge 정답률={sum(judge)/n:.3f}")
    print(f"일치율={agree/n:.3f} · Cohen's κ={cohens_kappa(human, judge):.3f}")
    print(f"judge 과대(사람0·judge1)={fp} · judge 과소(사람1·judge0)={fn}")

    print("\n--- 불일치 문항 ---")
    for h, j, it in pairs:
        if h != j:
            print(f"[사람{h}/judge{j}] Q: {it['question'][:70]}")
            print(f"    정답: {it['reference']}  ← 답변: {it['answer'][:60]}")


def main() -> None:
    """sample / score 서브커맨드를 파싱해 실행한다."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="사람 채점용 시트 생성")
    s.add_argument("--results", required=True)
    s.add_argument("--eval-set", default="config/eval_hotpot400.yaml")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="judge_labels.yaml")
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("score", help="채워진 시트로 일치도 산출")
    c.add_argument("--sheet", default="judge_labels.yaml")
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

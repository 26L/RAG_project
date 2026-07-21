"""기법 비교 통계 리포트 — 부트스트랩 95% CI + McNemar 쌍대검정.

왜 필요한가: 평균만 보면 "0.69 > 0.64" 를 우열로 착각한다. n=100 에선 CI 폭이 ±0.09 라
전부 중첩됐다(문제 1). 결론을 말하려면 CI 와 쌍대검정을 **항상 병기**해야 한다.

입력: --results 결과 JSON 경로들(라벨은 파일명) / --metric 채점 컬럼(기본 judge_correct)
출력: stdout — 기법별 평균+95% CI 표, 그리고 모든 쌍의 McNemar p값 표

사용: .venv/bin/python scripts/stats_report.py --results results/local_eval/*_hotpot400.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from itertools import combinations
from math import comb


def load(path: str, metric: str) -> tuple[str, list[float]]:
    """결과 파일에서 라벨과 문항별 채점값을 뽑는다(None 문항 제외 없이 0 처리).

    입력: path — 결과 JSON / metric — per_item 의 채점 키
    출력: (라벨, 문항별 값 리스트)
    """
    rows = json.load(open(path, encoding="utf-8"))["report"]["per_item"]
    label = os.path.splitext(os.path.basename(path))[0]
    return label, [float(r.get(metric) or 0.0) for r in rows]


def boot_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """부트스트랩 95% 신뢰구간(백분위법).

    입력: values — 문항별 값 / n_boot — 재표집 횟수 / seed — 재현용 시드
    출력: (하한, 상한)
    """
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def mcnemar_p(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """McNemar 정확검정(이항) — 같은 문항을 두 기법이 다르게 맞춘 쌍만 본다.

    입력: a, b — 두 기법의 문항별 0/1 채점(같은 순서·같은 길이)
    출력: (a만 맞음 수, b만 맞음 수, 양측 p값). 불일치가 없으면 p=1.0
    """
    n01 = sum(1 for x, y in zip(a, b) if x > y)  # a만 정답
    n10 = sum(1 for x, y in zip(a, b) if y > x)  # b만 정답
    n = n01 + n10
    if n == 0:
        return 0, 0, 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2 * tail)


def main() -> None:
    """결과 파일들을 읽어 CI 표와 McNemar 쌍대검정 표를 출력한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--metric", default="judge_correct")
    args = ap.parse_args()

    data = dict(load(p, args.metric) for p in args.results)
    n = len(next(iter(data.values())))

    print(f"지표: {args.metric} · n={n}\n")
    print(f"{'기법':32s} {'평균':>7s}  95% CI")
    print("-" * 60)
    for label, vals in sorted(data.items(), key=lambda kv: -sum(kv[1])):
        lo, hi = boot_ci(vals)
        print(f"{label:32s} {sum(vals)/len(vals):7.3f}  [{lo:.3f}, {hi:.3f}]")

    print(f"\n{'쌍대비교 (McNemar)':46s} {'a만':>4s} {'b만':>4s} {'p':>8s}")
    print("-" * 66)
    sig = 0
    for (la, va), (lb, vb) in combinations(data.items(), 2):
        n01, n10, p = mcnemar_p(va, vb)
        mark = " *" if p < 0.05 else ""
        print(f"{la[:22]:22s} vs {lb[:22]:22s} {n01:4d} {n10:4d} {p:8.4f}{mark}")
        sig += p < 0.05
    print(f"\n유의(p<0.05) 쌍: {sig} / {comb(len(data), 2)}")


if __name__ == "__main__":
    main()

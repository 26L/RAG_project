"""HotpotQA(공개 다중홉 벤치) 코퍼스 + 평가셋 생성 — 표본 수를 인자로 확대 가능.

왜 필요한가: n=100 은 Δ=0.08 을 검출할 검정력이 12~40% 뿐이라(McNemar 전부 무유의차)
기법 간 우열을 말할 수 없다. n=400 이면 Δ=0.08 검출력이 ~96% 로 올라간다.

재현성: distractor validation 스플릿을 **앞에서부터 순서대로** 잘라 쓰므로 n=400 의
앞 100 문항은 기존 n=100 과 완전히 동일하다(결과 비교 가능).

입력: --n 문항 수 / --data 코퍼스 출력 디렉토리 / --out 평가셋 YAML 경로
출력: 없음(코퍼스 .md 파일들 + 평가셋 YAML 을 씀. 문항·문서 수를 stdout 에 출력)

사용: .venv/bin/python scripts/build_hotpot_bench.py --n 400
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re

import yaml
from datasets import load_dataset

# answer_keywords 에서 뺄 기능어 — 포함 여부로 채점하므로 변별력이 없다.
_STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was"}


def doc_name(title: str) -> str:
    """위키 문서 제목을 코퍼스 파일명으로 바꾼다(제목 + md5 앞 6자리).

    입력: title — 위키 문단 제목
    출력: "Scott_Derrickson_d3a4c0.md" 형태의 파일명(제목 충돌 방지용 해시 접미)
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_")[:50]  # 파일명 길이 상한
    return f"{slug}_{hashlib.md5(title.encode()).hexdigest()[:6]}.md"


def keywords(answer: str) -> list[str]:
    """참조 정답에서 채점용 키워드(내용어)만 뽑는다.

    keyword_recall 은 부분문자열 매칭이라 구두점이 붙으면("Village,") 정상 답변도 놓친다
    → 문자·숫자만 남긴다(기존 n=100 셋의 버그를 여기서 교정. judge 가 주력이라 영향 작음).

    입력: answer — HotpotQA 정답 문자열
    출력: 기능어를 제외한 토큰 리스트(비면 정답 원문 1개)
    """
    toks = [t for t in re.findall(r"[0-9A-Za-z]+", answer) if t.lower() not in _STOP]
    return toks or [answer]


def main() -> None:
    """HotpotQA 앞 n문항으로 코퍼스와 평가셋을 만든다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--data", default="data/hotpot")
    ap.add_argument("--out", default="config/eval_hotpot.yaml")
    args = ap.parse_args()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    os.makedirs(args.data, exist_ok=True)

    written: set[str] = set()
    items = []
    for row in ds.select(range(args.n)):
        ctx = row["context"]
        # 질문마다 gold 2개 + distractor 8개 문단이 딸려온다 → 전부 코퍼스에 넣는다.
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            fname = doc_name(title)
            if fname in written:
                continue
            body = " ".join(sents).strip()  # 원 데이터가 문장 앞 공백을 가짐 → 기존 코퍼스와 동일 바이트
            path = os.path.join(args.data, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{body}\n")
            written.add(fname)

        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))  # 원 순서 유지
        items.append(
            {
                "question": row["question"],
                "type": "multi",  # HotpotQA 는 설계상 전부 2홉 다중홉
                "relevant_sources": [doc_name(t) for t in gold],
                "answer_keywords": keywords(row["answer"]),
                "reference_answer": row["answer"],
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(items, f, allow_unicode=True, sort_keys=False)

    print(f"문항 {len(items)} · 고유 문서 {len(written)} → {args.data}, {args.out}")


if __name__ == "__main__":
    main()

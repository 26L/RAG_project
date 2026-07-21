"""검색/답변 지표. 기법 무관 — QueryResult 의 contexts(출처)와 answer 만 사용한다.

지표 두 갈래: ① 자체 채점(hit·recall@k·precision·MRR·keyword_recall)
② 공개벤치 표준(EM·token F1) — 논문 수치와 직접 비교하기 위해 SQuAD 관례를 따른다.
"""
from __future__ import annotations

import re
from collections import Counter
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


# --- 공개벤치 표준 지표(EM/F1) — 논문 수치와 같은 자로 비교하기 위함 -------------
# SQuAD/HotpotQA 관례: 소문자화 + 관사 제거 + 구두점 제거 + 공백 정규화 후 비교.
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(text: str) -> str:
    """정답 비교용 정규화(SQuAD 표준) — 소문자·관사 제거·구두점 제거·공백 정리.

    입력: text — 원본 답변 문자열
    출력: 정규화된 문자열(비교·토큰화의 기준)
    """
    s = (text or "").lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str | None) -> float | None:
    """정규화 후 완전 일치면 1.0, 아니면 0.0 (공개벤치 EM).

    입력: prediction — 모델 답변 / gold — 참조 정답(없으면 채점 생략)
    출력: 1.0 / 0.0, gold 가 없으면 None
    """
    if not gold:
        return None
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str | None) -> float | None:
    """정규화 토큰 기준 F1 (공개벤치 표준). 부분 정답을 부분 점수로 인정한다.

    입력: prediction — 모델 답변 / gold — 참조 정답(없으면 채점 생략)
    출력: 0.0~1.0 F1, gold 가 없으면 None. 양쪽 다 빈 토큰이면 완전일치로 1.0
    """
    if not gold:
        return None
    p_tok = normalize_answer(prediction).split()
    g_tok = normalize_answer(gold).split()
    if not p_tok or not g_tok:
        return 1.0 if p_tok == g_tok else 0.0
    common = Counter(p_tok) & Counter(g_tok)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_tok)
    recall = n_same / len(g_tok)
    return 2 * precision * recall / (precision + recall)

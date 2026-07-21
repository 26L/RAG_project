"""LLM-as-judge — 모델 답변이 참조 정답과 사실적으로 일치하는지 0/1로 채점.

비용이 들므로(질문당 LLM 1회) eval 의 --judge 옵션에서만 사용한다.
"""
from __future__ import annotations

from typing import Any

_PROMPT = """다음 질의응답을 채점한다. (도메인·언어 무관 — 사실 일치만 본다)
질문, 참조 정답, 모델 답변이 주어진다. 모델 답변이 참조 정답과 핵심 사실에서 일치하면 1, 아니면 0만 출력하라. 다른 말은 하지 마라.

[질문]
{question}

[참조 정답]
{reference}

[모델 답변]
{answer}

판정(0 또는 1):"""


def judge_answer(judge_llm: Any, question: str, reference: str | None, answer: str) -> float | None:
    """일치하면 1.0, 불일치 0.0, 채점 불가(참조정답 없음/파싱실패) None.

    judge LLM 에 "0 또는 1만 출력" 프롬프트를 던지고 응답 텍스트에서 판정을 뽑는다.
    '1'만 있으면 1.0, '0'만 있으면 0.0, 둘 다 섞여 있으면 처음 나온 숫자를 채택한다.

    입력: judge_llm — complete(prompt) 를 가진 채점용 LLM / question — 질문
          reference — 참조 정답(없으면 채점 생략) / answer — 모델 답변
    출력: 1.0(정답) · 0.0(오답) · None(참조 정답이 없거나 응답에 0/1 이 없어 파싱 실패)
    """
    if not reference:
        return None
    prompt = _PROMPT.format(question=question, reference=reference, answer=answer)
    text = str(judge_llm.complete(prompt)).strip()
    if "1" in text and "0" not in text:
        return 1.0
    if "0" in text and "1" not in text:
        return 0.0
    # 혼재 시 첫 숫자 채택
    for ch in text:
        if ch in "01":
            return float(ch)
    return None

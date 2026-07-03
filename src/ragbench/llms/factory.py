"""설정에 따라 생성 LLM 을 만든다. 기본은 Google Gemini (CLAUDE.md §3)."""
from __future__ import annotations

import os
from typing import Any

from ..core.config import Config


def _google_api_key() -> str | None:
    # GEMINI_API_KEY 우선, 없으면 GOOGLE_API_KEY 사용 (둘 다 신형 SDK 가 인식).
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def build_llm(cfg: Config) -> Any:
    provider = cfg.llm.provider
    if provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI

        kwargs = dict(
            model=cfg.llm.model,
            api_key=_google_api_key(),
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        # thinking=False → thinking_budget 0 으로 사고 토큰 제거(추출·요약·채점 비용 절감).
        # 주의: generation_config 를 주면 GoogleGenAI 가 temperature/max_tokens 를 주입하는
        # 분기를 건너뛰므로, 여기서 명시해 재현성(temperature) 을 유지한다.
        if not getattr(cfg.llm, "thinking", True):
            from google.genai import types

            kwargs["generation_config"] = types.GenerateContentConfig(
                temperature=cfg.llm.temperature,
                max_output_tokens=cfg.llm.max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        return GoogleGenAI(**kwargs)

    if provider == "ollama":
        # 로컬/원격 Ollama. base_url 은 OLLAMA_BASE_URL 환경변수(기본 localhost:11434).
        # M2 등 다른 머신의 Ollama를 쓰려면 OLLAMA_BASE_URL=http://<ip>:11434 로 지정.
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=cfg.llm.model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=cfg.llm.temperature,
            request_timeout=600.0,
            context_window=cfg.llm.num_ctx,  # num_ctx 로 전달 — 기본 4096 절단 회피
        )

    if provider == "anthropic":
        # ANTHROPIC_API_KEY 환경변수에서 키를 읽는다.
        from llama_index.llms.anthropic import Anthropic

        return Anthropic(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )

    raise ValueError(f"지원하지 않는 LLM 공급자: {provider!r}")

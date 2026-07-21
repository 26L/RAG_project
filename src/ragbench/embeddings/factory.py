"""설정에 따라 임베딩 모델을 만든다 (교체 가능한 비교 축, CLAUDE.md §3).

새 임베딩 공급자는 여기에 분기를 추가한다. import 는 사용 시점에만 — 미설치 공급자가
전체 실행을 막지 않도록.
"""
from __future__ import annotations

import os
from typing import Any

from ..core.config import Config


def build_embed_model(cfg: Config) -> Any:
    """설정의 provider 에 맞는 임베딩 모델을 만든다.

    입력: cfg — provider("google"/"openai"/"local")·model 등 임베딩 설정
    출력: LlamaIndex 임베딩 객체 — google=GoogleGenAIEmbedding, openai=OpenAIEmbedding,
        local=HuggingFaceEmbedding(GPU 있으면 cuda). 그 외 provider 는 ValueError.
    """
    provider = cfg.embed.provider
    if provider == "google":
        # GEMINI_API_KEY 우선, 없으면 GOOGLE_API_KEY.
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return GoogleGenAIEmbedding(model_name=cfg.embed.model, api_key=api_key)

    if provider == "openai":
        # OPENAI_API_KEY 환경변수에서 키를 읽는다.
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=cfg.embed.model)

    if provider == "local":
        # 로컬 오픈소스 임베딩(BGE/E5 등). API 비용 0·rate limit 없음. GPU 자동 사용.
        import torch
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbedding(
            model_name=cfg.embed.model,
            device=device,
            embed_batch_size=8,  # 4GB VRAM 대비 보수적
        )

    raise ValueError(f"지원하지 않는 임베딩 공급자: {provider!r}")

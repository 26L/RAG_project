"""실험 설정 로딩. 기법/임베딩/청킹 파라미터는 모두 여기로 — 하드코딩 금지 (CLAUDE.md §7)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 1024
    num_ctx: int = 4096  # 입력 컨텍스트 창(Ollama). 기본 4096 — 온디바이스 제약
    thinking: bool = True  # Gemini thinking. 추출·요약·채점엔 불필요 → False로 비용/속도 절감


@dataclass
class EmbedConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 4
    data_dir: str = "data"
    storage_dir: str = "storage"
    extract_lang: str = "ko"  # 그래프 추출 프롬프트 언어("ko" 사내도메인 / "en" 범용). 외부벤치용.
    hipporag_embedding: str | None = None  # HippoRAG 임베딩(기본 facebook/contriever). e5는 OpenAI호환 서버.
    hipporag_embedding_base_url: str | None = None  # e5 OpenAI호환 서버 URL(공정비교). 이름에 "text-embedding" 필요.
    hipporag_llm_base_url: str | None = None  # HippoRAG 내부 LLM(OpenIE+트리플필터) 엔드포인트. 기본 Gemini. NIM 등 override.
    hipporag_llm_model: str | None = None     # 내부 LLM 모델. 기본 cfg.llm.model. NIM: meta/llama-3.3-70b-instruct
    hipporag_llm_key_env: str | None = None   # 내부 LLM API키 환경변수명. 기본 GEMINI_API_KEY. NIM: NVIDIA_API_KEY

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        data: dict = {}
        if path:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        llm = LLMConfig(**(data.get("llm") or {}))
        embed = EmbedConfig(**(data.get("embed") or {}))
        top = {k: v for k, v in data.items() if k not in ("llm", "embed")}
        return cls(llm=llm, embed=embed, **top)

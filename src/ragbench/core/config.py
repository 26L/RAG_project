"""실험 설정 로딩. 기법/임베딩/청킹 파라미터는 모두 여기로 — 하드코딩 금지 (CLAUDE.md §7)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    """생성 LLM 설정 — provider(공급자)·model(모델 ID)·temperature(무작위성)·max_tokens(출력 상한).

    num_ctx 는 Ollama 입력 컨텍스트 창, thinking 은 Gemini thinking 사용 여부.
    """
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 1024
    num_ctx: int = 4096  # 입력 컨텍스트 창(Ollama). 기본 4096 — 온디바이스 제약
    thinking: bool = True  # Gemini thinking. 추출·요약·채점엔 불필요 → False로 비용/속도 절감


@dataclass
class EmbedConfig:
    """임베딩 설정 — provider(공급자: openai/gemini/local 등)·model(임베딩 모델 ID)."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"


@dataclass
class Config:
    """실험 전체 설정 — LLM·임베딩 + 청킹(chunk_size/overlap)·검색(top_k)·경로(data_dir/storage_dir).

    extract_lang 은 그래프 추출 프롬프트 언어, hipporag_* 는 HippoRAG 어댑터 전용 override.
    """
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
        """YAML 파일에서 설정을 읽어 Config 를 만든다(없으면 기본값).

        입력: path — 설정 YAML 경로. None 이면 전부 기본값
        출력: llm/embed 는 중첩 dataclass 로, 나머지 키는 최상위 필드로 채운 Config
        """
        data: dict = {}
        if path:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        llm = LLMConfig(**(data.get("llm") or {}))
        embed = EmbedConfig(**(data.get("embed") or {}))
        top = {k: v for k, v in data.items() if k not in ("llm", "embed")}
        return cls(llm=llm, embed=embed, **top)

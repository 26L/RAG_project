"""문서 로드 (모든 기법이 공유). 청킹은 기법별 인덱싱 단계에서 설정한다."""
from __future__ import annotations

from typing import Any


def load_documents(data_dir: str) -> list[Any]:
    """data_dir 의 파일들을 LlamaIndex Document 리스트로 읽는다.

    하위 디렉토리까지 재귀 탐색한다. 청킹은 하지 않는다(기법별 인덱싱 단계 몫).

    입력: data_dir — 코퍼스 루트 경로
    출력: LlamaIndex Document 리스트(파일당 1개, 메타데이터에 파일명 포함)
    """
    from llama_index.core import SimpleDirectoryReader

    return SimpleDirectoryReader(data_dir, recursive=True).load_data()

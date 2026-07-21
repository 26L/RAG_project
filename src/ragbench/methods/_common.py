"""RAG 기법 공통 베이스 — LlamaIndex VectorStoreIndex 기반 인덱싱/로딩 공유.

검색 방식만 다른 백엔드(standard/bm25/hybrid)는 `_make_engine()` 만 구현한다.
"""
from __future__ import annotations

import os
import re
from typing import Any, Sequence

from ..core.config import Config
from ..core.interface import QueryResult, RagBackend, RetrievedContext

_TOKEN = re.compile(r"[A-Za-z0-9가-힣]+")

# 공개벤치(HotpotQA 등)용 짧은답 템플릿 — 정답이 2~3단어 span 이라 문장형 답변은
# EM/token-F1 이 구조적으로 깎인다(논문 수치와 비교 불가). answer_style="short" 로 전환.
#
# 근거 원칙: 추정 금지, 검색된 컨텍스트로 뒷받침되는 내용만 답한다(RAG 의 groundedness
# 관례 — 근거 없는 생성은 환각이므로 점수를 얻더라도 무효). 근거가 없으면 추측 대신
# 고정 토큰 NO_ANSWER 로 기권시켜, 오답과 기권을 분리 집계할 수 있게 한다.
_SHORT_QA_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the question using ONLY the context above. Do not use prior knowledge and "
    "do not speculate — every part of the answer must be supported by the context.\n"
    "Give the SHORTEST possible answer span (a name, entity, number, date, or yes/no). "
    "Output only the answer itself — no sentence, no explanation, no punctuation at the end.\n"
    "If the context does not support an answer, output exactly: NO_ANSWER\n"
    "Question: {query_str}\n"
    "Answer: "
)


def apply_answer_style(engine: Any, cfg: Config) -> Any:
    """cfg.answer_style 에 따라 질의 엔진의 답변 생성 템플릿을 교체한다.

    입력: engine — LlamaIndex 질의 엔진 / cfg — 설정(answer_style)
    출력: 같은 engine 객체("short" 면 text_qa_template 이 짧은답 템플릿으로 교체됨)
    """
    if getattr(cfg, "answer_style", "default") != "short":
        return engine
    from llama_index.core import PromptTemplate

    engine.update_prompts(
        {"response_synthesizer:text_qa_template": PromptTemplate(_SHORT_QA_TMPL)}
    )
    return engine


def ko_tokenize(text: str) -> list[str]:
    """한국어/영문/숫자 토큰화(어절·연속문자 단위). 형태소 분석 없이 BM25용 1차 토크나이저.

    입력: text — 원문 문자열
    출력: 소문자화된 토큰 리스트
    """
    return _TOKEN.findall(text.lower())


class LlamaIndexBackend(RagBackend):
    """VectorStoreIndex 로 인덱싱하고, 검색기만 바꿔 끼우는 공통 베이스."""

    name = "base"

    def __init__(self, cfg: Config, llm: Any, embed_model: Any):
        """백엔드를 초기화하고 LlamaIndex 전역 Settings(LLM·임베딩·청킹)를 세팅한다.

        입력: cfg — 설정(storage_dir·chunk_size·chunk_overlap·top_k), llm — 생성 LLM,
            embed_model — 임베딩 모델
        출력: 없음(인덱스 경로 self.persist_dir 결정, Settings 전역 설정)
        """
        self.cfg = cfg
        self.llm = llm
        self.embed_model = embed_model
        self.persist_dir = os.path.join(cfg.storage_dir, self.name)
        self._index = None

        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )

    def index(self, documents: Sequence[Any]) -> None:
        """문서를 청킹·임베딩해 VectorStoreIndex 를 만들고 디스크에 영속화한다.

        입력: documents — 로드된 LlamaIndex Document 목록
        출력: 없음(self._index 설정 + persist_dir 에 인덱스 저장)
        """
        from llama_index.core import VectorStoreIndex

        self._index = VectorStoreIndex.from_documents(list(documents))
        self._index.storage_context.persist(persist_dir=self.persist_dir)

    def _ensure_loaded(self) -> None:
        """인덱스가 메모리에 없으면 persist_dir 에서 읽어 올린다.

        출력: 없음(self._index 채움. 저장된 인덱스가 없으면 FileNotFoundError)
        """
        if self._index is not None:
            return
        if not os.path.isdir(self.persist_dir):
            raise FileNotFoundError(
                f"인덱스가 없습니다: {self.persist_dir}. 먼저 'ragbench index --method {self.name}' 를 실행하세요."
            )
        from llama_index.core import StorageContext, load_index_from_storage

        sc = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self._index = load_index_from_storage(sc)

    def _nodes(self) -> list[Any]:
        """인덱스 docstore 의 전체 청크 노드를 꺼낸다(BM25 등 비벡터 검색기 구성용).

        출력: 노드 리스트
        """
        return list(self._index.docstore.docs.values())

    def _make_engine(self) -> Any:
        """검색 방식을 정하는 서브클래스 확장점 — 질의 엔진을 만들어 반환한다.

        출력: LlamaIndex 질의 엔진(서브클래스가 구현. 베이스는 NotImplementedError)
        """
        raise NotImplementedError

    def query(self, question: str) -> QueryResult:
        """질문을 서브클래스의 엔진으로 검색·생성하고 공통 QueryResult 로 변환한다.

        입력: question — 자연어 질문
        출력: QueryResult — answer(생성 답변), contexts(source_nodes 를 RetrievedContext
            로 변환: 본문·file_name 출처·score·메타), metadata(method·top_k)
        """
        self._ensure_loaded()
        resp = apply_answer_style(self._make_engine(), self.cfg).query(question)
        contexts = [
            RetrievedContext(
                text=node.node.get_content(),
                source=node.node.metadata.get("file_name"),
                score=node.score,
                metadata=node.node.metadata,
            )
            for node in resp.source_nodes
        ]
        return QueryResult(
            answer=str(resp),
            contexts=contexts,
            metadata={"method": self.name, "top_k": self.cfg.top_k},
        )

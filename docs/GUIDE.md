# 가이드 — 실행 · 코드 지도 · 기술 레퍼런스

> 프로젝트를 **돌리고·고치고·찾아볼 때** 보는 문서. 결론은 [RESULT.md](RESULT.md).

---

## 실행 가이드

> ragbench 설치·실행·설정별 사용·트러블슈팅. 명령은 실제 검증된 것.

### 0. 사전 요구사항
| 항목 | 필요 | 비고 |
| --- | --- | --- |
| Python 3.10 + **uv** | 필수 | 패키지·가상환경 |
| GPU(CUDA) | 선택 | 로컬 임베딩(e5) 가속. 없으면 CPU |
| **Gemini API 키** | 클라우드 쓸 때 | `GEMINI_API_KEY` (ai.google.dev) |
| **Ollama** | 로컬 LLM 쓸 때 | gemma 등. `OLLAMA_BASE_URL` |
| **Docker** | 그래프 계열 쓸 때 | Neo4j 실행 |

### 1. 설치
```bash
uv venv --python 3.10 .venv
uv pip install -e .
```

### 2. 키 · 설정
```bash
cp .env.example .env      # GEMINI_API_KEY 등 채우기
```
`.env` 항목:
- `GEMINI_API_KEY=...` — Gemini 생성·임베딩·judge
- `OLLAMA_BASE_URL=http://<M2-IP>:11434` — 로컬 LLM(다른 머신이면 IP)
- `NEO4J_URL/USER/PASSWORD` — Neo4j(그래프 계열)

**설정 파일**(`config/*.yaml`) — 실행 시 `--config`로 선택:
| config | 생성 LLM | 임베딩 | 용도 |
| --- | --- | --- | --- |
| `default.yaml` | Gemini | Gemini | 기본(클라우드) |
| `ollama.yaml` | gemma(Ollama) | 로컬 e5 | **전부 로컬·무료** |
| `local.yaml` | Gemini | 로컬 e5 | 생성만 클라우드 |
| `gemini_extract.yaml` | Gemini(thinking off) | 로컬 e5 | 추출·생성·judge Gemini |

### 3. 기본 실행 3종 (index → query → eval)
```bash
.venv/bin/ragbench index --method hybrid --config config/ollama.yaml --data data/company

.venv/bin/ragbench query --method hybrid --config config/ollama.yaml "연차는 어떤 규정에 근거하나?"

.venv/bin/ragbench eval  --method hybrid --config config/ollama.yaml \
    --eval-set config/eval_sample.yaml --judge
```
- `--method`: `standard` `bm25` `hybrid` `graphrag` `graphrag_e2b` `graphrag_e2b_l5` `graphrag_e2b_adaptive` `graphrag_e2b_hybrid`
- `--judge`: LLM-as-judge 정답 채점(config의 llm 사용)
- `--data`: 코퍼스 폴더 (기본 `data/company`)

### 4. 설정별 실행

#### (A) 전부 로컬·무료 (gemma + e5)
```bash
export OLLAMA_BASE_URL=http://<M2-IP>:11434
.venv/bin/ragbench index --method standard --config config/ollama.yaml --data data/company
.venv/bin/ragbench eval  --method standard --config config/ollama.yaml --eval-set config/eval_sample.yaml --judge
```

#### (B) Gemini (추출·생성·judge) + 로컬 e5
```bash
.venv/bin/ragbench eval --method hybrid --config config/gemini_extract.yaml \
    --eval-set config/eval_sample.yaml --judge
```

### 5. 그래프 계열 실행 (graphrag_e2b*)
```bash
docker compose up -d                 # http://localhost:7474

export OLLAMA_BASE_URL=http://<M2-IP>:11434
.venv/bin/ragbench index --method graphrag_e2b --config config/ollama.yaml --data data/company

.venv/bin/python scripts/migrate_graph_to_neo4j.py storage/graphrag_e2b

.venv/bin/python scripts/build_community_summaries.py config/ollama.yaml storage/graphrag_e2b

.venv/bin/ragbench eval --method graphrag_e2b_hybrid --config config/ollama.yaml \
    --eval-set config/eval_sample.yaml --judge
```
> `graphrag_e2b_hybrid`는 **standard 인덱스도 필요**(직접 벡터검색용) — 먼저 `index --method standard` 실행.

### 6. 결과 비교
```bash
.venv/bin/ragbench compare               # results/*.json 라벨 비교표
```
- 결과 파일: `results/<method>.json`, `results/local_eval/*.json` (`per_item` 포함 → 재채점 가능)

### 7. 트러블슈팅 (실제 겪은 이슈)
| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `MAX_TOKENS` 조기 종료 | gemini-2.5 thinking이 출력 토큰 소모 | config `max_tokens` ↑ (그래프 추출은 8192) |
| 입력이 4096에서 잘림 | Ollama 기본 `num_ctx=4096` | config `llm.num_ctx: 8192` |
| 임베딩 429(rate limit) | Gemini 임베딩 분당 한도 | **로컬 e5로 교체**(config embed: local) |
| 그래프 인덱싱 중첩 async 에러 | PropertyGraphIndex 검색기 | `nest_asyncio.apply()` (코드 내장됨) |
| Gemini 비용↑·느림 | thinking 켜짐 | config `llm.thinking: false` |
| 그래프 recall 낮음 | 재랭커 트리플 오염·단일 검색 | 재랭커 수정 + `graphrag_e2b_hybrid` 사용 |
| M2 gemma 매우 느림 | 온디바이스 8GB | 청크↑·문항↓, 또는 Gemini(클라우드) |

### 8. 빠른 시작 (한 번에)
```bash
uv venv --python 3.10 .venv && uv pip install -e .
cp .env.example .env                                  # 키 채우기
docker compose up -d                                  # (그래프 쓸 때)
.venv/bin/ragbench index --method hybrid --config config/ollama.yaml --data data/company
.venv/bin/ragbench eval  --method hybrid --config config/ollama.yaml --eval-set config/eval_sample.yaml --judge
```

> 자세한 코드 위치는 [GUIDE.md](GUIDE.md), 기술 문서는 [GUIDE.md](GUIDE.md).

---

## 코드 지도 (아키텍처)

> `ragbench` 코드베이스 탐색·수정용 가이드. **"무슨 코드가 어디 있고, X를 고치려면 어디를 보는지"** 에 초점. 설계 배경·실험 결과는 [CLAUDE.md](../CLAUDE.md), 진행 타임라인은 [LOG.md](LOG.md) 참고.

### 1. 한눈에 — 플러그형 설계

모든 RAG 기법을 **공통 인터페이스**(`RagBackend`) 뒤에 두고, 임베딩·LLM을 교체 축으로 둔다. 평가 하니스는 기법을 몰라도 된다.

```text
  CLI (cli.py)  ──build_llm/build_embed_model──►  LLM·임베딩 백엔드
       │                                          (llms/, embeddings/)
       │ build_backend(name, cfg, llm, embed)
       ▼
  registry.py  ──►  RagBackend 구현 (methods/)
       │                 · index(documents) -> 영속 인덱스
       │                 · query(question)  -> QueryResult{answer, contexts, metadata}
       ▼
  eval/harness.py  ──►  지표 집계 (metrics·judge)  ──►  results/*.json
```

- **핵심 계약**: [core/interface.py](../src/ragbench/core/interface.py) — `RagBackend`(ABC), `QueryResult`, `RetrievedContext`
- **기법 등록**: [registry.py](../src/ragbench/registry.py) — 이름→클래스 매핑(`METHODS` dict). **새 기법은 여기 한 줄 + methods/ 어댑터**.

### 2. 요청 흐름 (3가지 명령)

CLI 진입점: [cli.py](../src/ragbench/cli.py) `main()` → 서브커맨드 `index` / `query` / `eval` / `compare`. 콘솔스크립트 `ragbench`([pyproject.toml](../pyproject.toml)).

| 명령 | 흐름 | 주요 파일 |
| --- | --- | --- |
| **index** | 문서 로드 → 기법.index() → 저장 | `cmd_index` → [ingest/loader.py](../src/ragbench/ingest/loader.py) → `methods/*` |
| **query** | 기법.query() → 답변+출처 | `cmd_query` → `methods/*._make_engine()` |
| **eval** | 평가셋 로드 → 각 문항 query → 지표 집계 | `cmd_eval` → [eval/harness.py](../src/ragbench/eval/harness.py) |

**공통 백엔드 골격**: [methods/_common.py](../src/ragbench/methods/_common.py) `LlamaIndexBackend` — `persist_dir`, `index()`/`query()` 기본 구현, `ko_tokenize`(한국어 BM25 토크나이저). 대부분의 기법이 이걸 상속.

### 3. 디렉토리 지도

```text
src/ragbench/
  core/
    interface.py   ★ RagBackend(ABC)·QueryResult·RetrievedContext — 모든 기법의 계약
    config.py      ★ Config·LLMConfig·EmbedConfig (YAML 로딩, Config.load)
  ingest/loader.py   문서 로드 (SimpleDirectoryReader 래핑)
  embeddings/factory.py  build_embed_model — google / openai / local(e5) 분기
  llms/factory.py        build_llm — google(Gemini) / anthropic / ollama 분기
  methods/
    _common.py     LlamaIndexBackend(공통 골격)·ko_tokenize
    standard.py    StandardRAG (의미 검색, VectorStoreIndex)
    bm25.py        BM25RAG (키워드)
    hybrid.py      HybridRAG (벡터+BM25 RRF)
    graphrag.py    GraphRAG·GraphRAGSchema·GraphRAGDynamic (추출기 3종 비교)
    graphrag_e2b.py  ★ E2B 계열 (아래 §5) — 이 프로젝트 핵심
  eval/
    dataset.py     EvalItem·load_eval_set (평가셋 YAML 파싱)
    harness.py     run_eval — 문항별 query + 지표 집계 + per_item 저장
    metrics.py     retrieval_metrics(recall@k·precision·mrr)·keyword_recall
    judge.py       judge_answer (LLM-as-judge 0/1)
  registry.py      ★ METHODS dict (이름→기법)·build_backend
  cli.py           index/query/eval/compare 진입점
config/            *.yaml (기법·임베딩·모델·청킹·top-k·평가셋)
scripts/           코퍼스 생성·Neo4j 적재·커뮤니티 요약
data/·storage/·results/  코퍼스·인덱스·결과 (git 제외)
```

### 4. 핵심 계약 — RagBackend

[core/interface.py](../src/ragbench/core/interface.py):

```python
class RagBackend(ABC):
    name: str
    def index(self, documents) -> None: ...      # 구축·저장
    def query(self, question) -> QueryResult: ...  # {answer, contexts, metadata}
```

- `QueryResult.contexts`: `RetrievedContext[]` (출처 청크/노드) — 평가의 recall@k 계산에 쓰임.
- 새 기법은 이 계약만 지키면 평가 하니스·CLI가 그대로 동작.

### 5. graphrag_e2b 계열 (가장 복잡 — 상세)

[methods/graphrag_e2b.py](../src/ragbench/methods/graphrag_e2b.py). **같은 그래프를 쓰고 "검색 방식"만 다른** 4개 기법 + 추출·정규화 유틸.

#### 5.1 추출 (그래프 구축)
- `parse_extraction()` — LLM 산문("이름|유형|설명") → (엔티티, 관계) 파싱
- `normalize_name/type/rel()` — 정규화(「」 병합·타입 교정·관계 표면형 통합)
- `_build_extractor_cls()` → `E2BExtractor` — 청크마다 LLM 호출해 EntityNode+Relation 생성. `index()`는 부모 `GraphRAG.index()`(PropertyGraphIndex) 사용.

#### 5.2 검색 (기법별로 다름 — `_make_engine`)
| 클래스 | 검색 방식 | 비고 |
| --- | --- | --- |
| `GraphRAGE2B` | 그래프 검색 + `E5Rerank` 재순위 | 기준 |
| `GraphRAGE2BL5` | 위 + 커뮤니티 요약 주입 | `_load_summaries()`(community_summaries.json), persist_dir=`storage/graphrag_e2b` 재사용 |
| `GraphRAGE2BAdaptive` | 위 + 질의 유형 라우팅 | `_GraphCommunityRetriever._inject_summaries`(router LLM) |
| **`GraphRAGE2BHybrid`** | **그래프 + 직접 청크벡터 RRF 융합** | `standard` 인덱스 재사용, `QueryFusionRetriever` |

#### 5.3 검색 보조 (팩토리로 생성 — import 순서 회피)
- `_build_reranker_cls()` → `E5Rerank` — 질의-노드 e5 유사도 재순위. **`clean()`**: PropertyGraphIndex가 붙이는 트리플 접두사 제거 후 임베딩(e5 512토큰 오염 버그 수정 — CLAUDE.md §10.5).
- `_build_community_retriever_cls()` → `_GraphCommunityRetriever` — 그래프검색+재순위+커뮤니티요약 결합, 라우터로 주입 판정.

### 6. "이걸 고치려면 여기를 보라" (작업→파일 지도)

| 하고 싶은 것 | 볼 곳 |
| --- | --- |
| **새 RAG 기법 추가** | `methods/` 에 `RagBackend`(또는 `LlamaIndexBackend`) 어댑터 + [registry.py](../src/ragbench/registry.py) `METHODS` 한 줄 |
| **임베딩 백엔드 추가/교체** | [embeddings/factory.py](../src/ragbench/embeddings/factory.py) `build_embed_model` 분기 + config `embed.provider/model` |
| **생성 LLM 추가/교체** | [llms/factory.py](../src/ragbench/llms/factory.py) `build_llm` 분기 (google/anthropic/ollama) |
| **Gemini thinking 끄기(비용)** | config `llm.thinking: false` → factory google 분기 |
| **청킹·top-k·모델 변경** | `config/*.yaml` (하드코딩 금지 — CLAUDE.md §7) |
| **그래프 추출 품질(엔티티·관계)** | graphrag_e2b.py `parse_extraction`·`normalize_*`·`E2BExtractor` |
| **그래프 검색/재순위 개선** | graphrag_e2b.py `_make_engine`·`E5Rerank`·`_GraphCommunityRetriever` |
| **평가 지표 추가/수정** | [eval/metrics.py](../src/ragbench/eval/metrics.py)·[eval/harness.py](../src/ragbench/eval/harness.py) |
| **LLM-judge 채점 로직** | [eval/judge.py](../src/ragbench/eval/judge.py) |
| **평가셋(문항·정답) 편집** | [config/eval_sample.yaml](../config/eval_sample.yaml) (`question·type·relevant_sources·answer_keywords·reference_answer`) |
| **CLI 명령/옵션** | [cli.py](../src/ragbench/cli.py) |

### 7. 설정 시스템

[core/config.py](../src/ragbench/core/config.py) — `Config.load(path)`가 YAML→dataclass. 필드: `llm`(provider/model/temperature/max_tokens/num_ctx/thinking), `embed`(provider/model), `chunk_size`·`chunk_overlap`·`top_k`·`data_dir`·`storage_dir`.

| config | 용도 |
| --- | --- |
| `default.yaml` | Gemini 생성+임베딩 (기본) |
| `ollama.yaml` | 로컬 — gemma(Ollama)+e5, chunk 1024, num_ctx 8192 |
| `local.yaml` | Gemini 생성 + 로컬 e5 |
| `gemini_extract.yaml` | Gemini 추출·생성·judge (thinking off) + 로컬 e5 |
| `eval_sample.yaml` / `eval_global_only.yaml` | 평가셋 |

### 8. 저장소·데이터 레이아웃 (git 제외)

- `data/company/` — 코퍼스(가상 회사 문서). `data/noise/` — 노이즈 토글.
- `storage/<method>/` — 기법별 영속 인덱스. graphrag 계열: `property_graph_store.json`(노드·관계)·`default__vector_store.json`(임베딩)·`docstore.json`(청크)·`community_summaries.json`(L5). `_gemma_v2`·`_gemini_sub` 등은 비교용 백업.
- `results/` · `results/local_eval/` — 평가 결과 JSON(`per_item` 포함).

### 9. 외부 인프라

| 서비스 | 용도 | 설정 |
| --- | --- | --- |
| **Ollama** (M2 등) | 로컬 LLM(gemma) | `OLLAMA_BASE_URL` 환경변수 |
| **Neo4j + GDS** | 그래프 시각화·커뮤니티(Louvain)·PageRank | [docker-compose.yml](../docker-compose.yml) (`docker compose up -d`) |
| **Google AI Studio** | Gemini 생성·임베딩·judge | `GEMINI_API_KEY` ([.env](../.env.example)) |

관련 스크립트: [scripts/migrate_graph_to_neo4j.py](../scripts/migrate_graph_to_neo4j.py)(그래프→Neo4j 적재), [scripts/build_community_summaries.py](../scripts/build_community_summaries.py)(GDS Louvain+LLM 요약), [scripts/generate_rich_corpus.py](../scripts/generate_rich_corpus.py)(코퍼스 생성).

---

## 기술 레퍼런스 (공식 문서·링크)

> ragbench가 쓰는 기술의 **공식 가이드·레포** 모음. 혼자 빌드/학습 시 1차 근거.
> ⚠️ URL·패키지 버전은 변할 수 있으니 접속 시 최신판 확인.

### 1. 프레임워크 핵심 — LlamaIndex
- 공식 문서: https://docs.llamaindex.ai
- GitHub: https://github.com/run-llama/llama_index
- 핵심 개념 페이지(문서에서 검색):
  - `VectorStoreIndex` — 벡터 색인·검색 (standard)
  - `PropertyGraphIndex` — 지식그래프 색인 (graphrag)
  - `QueryFusionRetriever` — 여러 retriever RRF 융합 (hybrid)
  - `BaseRetriever` / `BaseNodePostprocessor` — 커스텀 검색·재순위
  - `TransformComponent` — 커스텀 추출기(E2B)
  - `StorageContext`, `load_index_from_storage` — 영속화
  - `SimpleDirectoryReader`, `SentenceSplitter` — 로드·청킹

### 2. LLM · 임베딩 (LlamaIndex 통합 패키지)
| 용도 | 패키지 | 참고 |
| --- | --- | --- |
| Google Gemini 생성 | `llama-index-llms-google-genai` | https://docs.llamaindex.ai (검색: GoogleGenAI) |
| Google 임베딩 | `llama-index-embeddings-google-genai` | 〃 |
| Ollama(로컬) 생성 | `llama-index-llms-ollama` | 〃 |
| Anthropic 생성 | `llama-index-llms-anthropic` | 〃 |
| HuggingFace 임베딩(e5) | `llama-index-embeddings-huggingface` | 〃 |
| BM25 검색 | `llama-index-retrievers-bm25` | 〃 |
| Neo4j 그래프 저장 | `llama-index-graph-stores-neo4j` | 〃 |

**외부 서비스 문서**
- Google Gemini API: https://ai.google.dev/gemini-api/docs (모델·가격·thinking·API키)
- Ollama: https://ollama.com · GitHub https://github.com/ollama/ollama (로컬 LLM 실행·`/api`)
- 임베딩 모델 e5: https://huggingface.co/intfloat/multilingual-e5-small

### 3. 그래프 — Neo4j + GDS
- Neo4j 공식 문서: https://neo4j.com/docs
- Graph Data Science(GDS, 커뮤니티·PageRank): https://neo4j.com/docs/graph-data-science
- GDS GitHub: https://github.com/neo4j/graph-data-science
- Cypher(쿼리 언어): https://neo4j.com/docs/cypher-manual
- Docker 이미지: https://hub.docker.com/_/neo4j
- 핵심 알고리즘: `gds.louvain`(커뮤니티, L5), `gds.pageRank`(HippoRAG류)

### 4. 개발 도구
- uv(패키지·가상환경): https://docs.astral.sh/uv · GitHub https://github.com/astral-sh/uv
- Docker Compose: https://docs.docker.com/compose

### 5. 논문 · 공식 구현 (검색 방법의 진화)
| 기법 | 논문(arXiv) | 공식 구현 |
| --- | --- | --- |
| RAG (원조) | https://arxiv.org/abs/2005.11401 | HuggingFace Transformers `examples/rag` |
| **GraphRAG** | https://arxiv.org/abs/2404.16130 | https://github.com/microsoft/graphrag |
| CausalRAG | https://arxiv.org/abs/2503.19878 | https://github.com/Pwnb/CausalRAG |
| LightRAG | https://arxiv.org/abs/2410.05779 | https://github.com/HKUDS/LightRAG |
| HippoRAG / HippoRAG2 | https://arxiv.org/abs/2405.14831 · https://arxiv.org/abs/2502.14802 | https://github.com/OSU-NLP-Group/HippoRAG |
| LeanRAG | https://arxiv.org/abs/2508.10391 | https://github.com/RaZzzyz/LeanRAG |

### 6. 혼자 빌드 순서 (파일 → 문서 매핑)
```
1. loader/config/interface  →  LlamaIndex 시작(SimpleDirectoryReader·SentenceSplitter)
2. standard + eval 하니스    →  VectorStoreIndex        ← 여기까지면 RAG 이해 완료
3. bm25 → hybrid             →  BM25Retriever·QueryFusionRetriever
4. graphrag                  →  PropertyGraphIndex·*PathExtractor
5. 커스텀 추출기·재랭커      →  TransformComponent·BaseNodePostprocessor
6. Neo4j·커뮤니티요약        →  Neo4jPropertyGraphStore·GDS Louvain
```
> **2번(standard+eval)까지 혼자** 해보면 "RAG 짤 수 있다" 확신이 생김. 나머지는 그 위에 검색 방식만 교체.

---

## 논문 정리 — GraphRAG

> Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — arXiv [2404.16130](https://arxiv.org/abs/2404.16130) / 공식 구현 [microsoft/graphrag](https://github.com/microsoft/graphrag). 구현 시 1차 근거(§7 규칙). (정리: 2026-06-30)

### 1. 핵심 문제의식
일반 벡터 RAG는 **"이 문서 전체의 주요 주제는?"** 같은 **전역(global)·종합형 질문**에 약하다. 이는 검색 문제가 아니라 **질의 중심 요약(query-focused summarization)** 문제이기 때문이다. GraphRAG는 **지식그래프 + 계층적 커뮤니티 요약**으로 코퍼스 전체에 대한 "의미 파악(sensemaking)"을 가능하게 한다.

### 2. 인덱싱 파이프라인
1. **청킹**: 문서를 **600토큰 청크(100토큰 겹침)**로 분할. 청크가 크면 LLM 호출 수↓·비용↓이지만 청크 앞부분 정보의 recall이 떨어짐.
2. **LLM 추출** — 세 가지:
   - **엔티티**: 이름·유형·**설명**(예: "NeoChip"(회사) + 설명)
   - **관계**: *명확히 관련된* (출발, 도착) 엔티티 쌍 + **관계 설명**
   - **주장(claim)**: 검증 가능한 사실 진술
   - 도메인별 **few-shot 예시**로 in-context learning
   - **Self-reflection**(누락 엔티티 재추출): 600토큰 청크에서 2400토큰 대비 엔티티를 거의 **2배** 더 검출
3. **그래프 구축**: 엔티티→노드, 관계→엣지(중복 횟수=가중치). 노드/엣지 **설명을 집약·요약**. 엔티티 매칭은 정확 문자열 일치(soft matching도 가능).
4. **계층적 커뮤니티 탐지**: **Leiden 알고리즘**을 재귀 적용 → 더 못 나눌 때까지 하위 커뮤니티 분할(각 레벨은 상호배타·전체포괄).
5. **커뮤니티 요약**: 리프 커뮤니티부터(노드 차수 합 내림차순) 토큰 한도까지 채워 요약. 상위 레벨은 하위 요약으로 치환. 요약은 **제목·요약·중요도(0~10)·상세 발견(5~10개)** 구조.

### 3. 질의 파이프라인
- **전역 검색(Map-Reduce)**:
  - **Map**: 커뮤니티 요약을 섞어 청크로 나눠 **병렬 부분 답변** + 각 답변에 **유용성 점수(0~100)**
  - **Reduce**: 점수 내림차순으로 한도까지 모아 **최종 답변** 합성
- **커뮤니티 레벨 C0~C3**: C0(루트, 가장 적음) ~ C3(최하위, 가장 많음) 중 선택해 질의.

### 4. 주요 파라미터
| 항목 | 값 |
| --- | --- |
| 청크 크기 / 겹침 | 600 / 100 토큰 |
| 커뮤니티 요약·답변 컨텍스트 | 각 8k 토큰 |
| 평가 질문 생성 | K=M=N=5 → 125문항 |

### 5. 평가 · 결과
- **데이터셋**: 팟캐스트(~100만 토큰), 뉴스(~170만 토큰)
- **지표**: 종합성(comprehensiveness)·다양성(diversity)·역량강화(empowerment) + 대조용 직접성(directness) — **LLM-as-judge 승률**
- **결과(vs 벡터 RAG)**: 종합성 승률 **72~83%**, 다양성 **62~82%** (p<.01~.001). C0(루트) 사용 시 원문 요약 대비 토큰 **9~43배**(뉴스 97%) 절감.

### 6. 우리 프로젝트 시사점
- GraphRAG의 강점은 **전역·종합형 질문**(전체 주제·경향). 우리 평가셋은 대부분 **핀포인트 사실·다중 홉** → GraphRAG의 강점이 드러나기 어려움 → 평가셋에 **전역(global) 문항 추가** 필요(반영 완료, 8문항).
- 핵심 구성요소(엔티티 **설명**, 관계 **설명**, **커뮤니티 요약**)가 우리 구현엔 빠져 있었음(CLAUDE.md §10, [DESIGN.md](DESIGN.md)).

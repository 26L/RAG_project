# ragbench — RAG 방법론 비교·검증 벤치마크

> 여러 RAG 기법과 임베딩을 **같은 코퍼스·질문셋·조건**으로 돌려, "문서 간 연결성" 문제에 **어떤 기법이 어떤 질문 유형에 강한지**를 데이터로 검증하는 벤치마크.

핵심 질문: *파편적으로 흩어진 문서들에서, 불필요한 정보가 아니라 정확한 근거로 답하려면 어떤 검색·생성 방식이 좋은가?*

---

## 📌 프로젝트 개요

- **문제**: 사내 문서처럼 서로 참조·연결된 문서 집합에서, 단일 사실·다중 홉·관계·전역 질문에 정확히 답하는 RAG를 찾는다.
- **접근**: 모든 기법을 **공통 인터페이스**(`index`/`query`) 뒤에 두고, 임베딩을 교체 가능한 축으로 두어, **동일 평가 하니스**로 비교한다.
- **차별점**: 단순 recall@k를 넘어 **LLM-as-judge**(답변 정답성)와 **질문 유형별**(single/multi/relational/global) 분해로, 그래프 계열에 불리한 지표 편향까지 통제한다.

---

## 🎯 핵심 결과

![결과 비교 — 그래프 검색만 0.333 → 그래프+직접벡터 0.806](docs/img/results.svg)

> 인터랙티브 버전(검증 여정 5단계 포함): Claude Artifact — 비공개 링크(소유자만).

**최종 비교** (동일 조건 — 210문서·36문항·Gemini 그래프·생성·LLM-judge)

| 방식 | judge(정답률) | single | multi | relational | global |
| --- | --- | --- | --- | --- | --- |
| **standard (평면-의미)** | **0.833** | 1.00 | 0.92 | 1.00 | 0.38 |
| hybrid (평면-의미+키워드) | 0.778 | 1.00 | 0.92 | 0.80 | 0.25 |
| graphrag_e2b (**그래프만**) | 0.333 | 0.36 | 0.50 | 0.20 | 0.12 |
| graphrag_e2b_l5 (그래프+커뮤니티요약) | 0.389 | 0.45 | 0.42 | 0.40 | 0.25 |
| **★ graphrag_e2b_hybrid (그래프+직접벡터)** | **0.806** | 1.00 | 0.92 | 0.80 | 0.38 |

### 핵심 발견

1. **GraphRAG 저성능은 방법론의 구조적 한계가 아니었다** — 5단계 검증(강judge 재채점 → 강추출 재구축 → **재랭커 임베딩 버그** 발견·수정 → 단일 검색 기법 진단 → **검색층 하이브리드**)으로 원인을 규명했다. 원인은 **① 재랭커 임베딩 버그**(검색 노드의 `트리플+원문`을 e5가 앞부분만 임베딩해 정답 청크 탈락) **+ ② 단일 검색 기법**(그래프 검색만 사용).
2. **올바른 검색층이면 그래프가 평면을 따라잡는다** — 그래프 검색에 **직접 청크 벡터검색을 RRF로 융합**하니 **judge 0.333 → 0.806**(2.4배), 핀포인트(single 1.0·multi 0.92)가 평면 수준으로 회복되면서 **전역(global 0.38)·연결성 강점은 유지**. 의미+키워드=하이브리드가 단일을 이겼던 것과 같은 원리(그래프+벡터).
3. **벤치마크 함정** — 약한 judge·단일 검색·미세 버그가 결론을 통째로 왜곡할 수 있다. **강judge 교차검증 + 실제 LLM 전달 컨텍스트 검사**로 "구조적 한계"라는 오결론을 뒤집었다.

> 전체 검증 여정(5단계)과 수치는 [`CLAUDE.md`](CLAUDE.md)(§10.5)와 [`docs/WORKLOG.md`](docs/WORKLOG.md) 참고.

---

## ⚠️ 실험 전제 · 한계 (결과 해석 전 필독)

**이 벤치마크는 "이상적 상황"에서 수행됐다. 아래 결과는 실제 환경의 상한값(best-case)이다.**

- **가상(합성) 코퍼스** — 실제 사내문서가 아니라 가상 회사 「주식회사 하울」 문서다.
- **인용이 깨끗하다** — 연결성 테스트를 위해 문서들이 서로를 **「문서명」으로 명시적·결정론적으로 인용**하도록 설계했다. 실제 문서의 참조는 지저분하다(암시적·표현 다양·부분 인용·오타).
- **따라서 그래프의 연결성 이득 · L4(문서참조 그래프) 효과는 낙관적(상한)** — 깨끗한 인용이 만든 최선값이며, 실제 문서에선 **갭(하락)** 이 생긴다. 절대 정답률 수치도 이상적 코퍼스 기준이다.
- **단, 상대적·메커니즘 결론은 견고하다** — LLM-judge는 문서명이 아니라 **사실**을 채점(36/36 사실 기반 정답)하고, "검색층 하이브리드가 단일 검색을 이긴다"는 **검색 메커니즘** 결론이라 코퍼스 성격과 무관하다.
- **실무 적용 전 필수** — 실제(지저분한) 문서로 재검증해야 한다.

---

## 🏗️ 아키텍처

```text
        공통 코퍼스 (data/)                    공통 평가셋 (config/eval_*.yaml)
              │                                          │
              ▼                                          ▼
     [임베딩 백엔드 N개]                          ┌─────────────────┐
              │                                   │  평가 하니스     │
              ▼                    ── 동일 질의 ──│ recall@k·MRR·   │
   ┌──────────────────────────┐   ── 답변+출처 ─▶│ judge·유형분해   │
   │ RAG 백엔드 (공통 인터페이스)│                 └─────────────────┘
   │  index(corpus)/query(q)   │
   └──────────────────────────┘
     ▲    ▲    ▲    ▲    ▲
  standard bm25 hybrid graphrag graphrag_e2b(_l5)  ← 각각 같은 계약 구현
```

- **공통 계약**: `index(documents)` / `query(question) -> {answer, contexts, metadata}`
- 새 기법은 `methods/`에 어댑터 추가 + `registry.py` 한 줄. **평가 하니스는 기법을 몰라도 된다.**

---

## 🧪 비교 대상

| 기법 | 개념 | 상태 |
| --- | --- | --- |
| Standard RAG | 벡터 검색 → top-k 청크 → 생성 | ✅ |
| BM25 | 키워드(렉시컬) 검색 | ✅ |
| Hybrid | 벡터 + BM25 (RRF 융합) | ✅ |
| GraphRAG | 지식그래프 + 그래프 탐색 | ✅ |
| **graphrag_e2b** | 로컬 산문추출로 타입+설명 채운 그래프 | ✅ |
| **graphrag_e2b_l5** | 위 + 커뮤니티 탐지·요약(전역 강화) | ✅ |
| LightRAG / HippoRAG2 / … | 듀얼레벨 / PageRank 등 | 🔜 |

임베딩 축: 로컬 `multilingual-e5-small`(기본), Gemini `gemini-embedding-001`, OpenAI 등 교체 가능.

---

## ⚙️ 기술 스택

**Python · LlamaIndex** · 로컬 LLM(**Ollama / Gemma**) · 로컬 임베딩(**e5**, GPU) · **Neo4j + GDS**(커뮤니티·PageRank·시각화) · BM25 · uv

> 전 과정 **로컬·무료** 실행 가능(API 비용 0). 생성/임베딩을 클라우드(Gemini·OpenAI)로 교체하는 축도 지원.

---

## 🚀 설치 & 실행

```bash
# 1) 환경
uv venv --python 3.10 .venv
uv pip install -e .

# 2) 키/설정 (클라우드 사용 시)
cp .env.example .env     # GEMINI_API_KEY 등

# 3) (선택) 로컬 그래프 스택 — Neo4j
docker compose up -d     # http://localhost:7474

# 4) 인덱싱 → 질의 → 평가
.venv/bin/ragbench index --method hybrid --config config/ollama.yaml --data data/company
.venv/bin/ragbench query --method hybrid --config config/ollama.yaml "연차는 어떤 규정에 근거하나?"
.venv/bin/ragbench eval  --method hybrid --config config/ollama.yaml --eval-set config/eval_sample.yaml --judge
```

기법·임베딩·모델·청킹·top-k는 `config/*.yaml`에서 교체(하드코딩 없음).

---

## 📁 프로젝트 구조

```text
src/ragbench/
  core/         # 공통 인터페이스·설정
  ingest/       # 문서 로딩
  embeddings/   # 임베딩 백엔드 팩토리
  llms/         # 생성 LLM 팩토리 (google/anthropic/ollama)
  methods/      # 기법 어댑터 (standard·bm25·hybrid·graphrag·graphrag_e2b…)
  eval/         # 평가 하니스 (metrics·judge·harness)
  registry.py   # 이름 → 기법 매핑
  cli.py        # index / query / eval
config/         # 기법·임베딩·청킹·평가셋 설정
scripts/        # 코퍼스 생성·Neo4j 적재·커뮤니티 요약
docs/           # 설계·논문정리·작업 타임라인
```

---

## 📊 평가 방법론

- **공정 비교**: 모든 기법이 같은 코퍼스·질문셋·생성 LLM·임베딩·청크를 쓰고, **기법만 변수**로 둔다.
- **지표**: 검색(recall@k·precision@k·MRR) + 답변(keyword_recall·**LLM-judge**) + 비용(지연).
- **질문 유형 분해**: single(단일 사실) / multi(다중 홉) / relational(관계) / global(전역·종합) — 기법별 강점 영역을 분리 측정.
- **재현성**: 설정은 `config/`, 결과는 `results/`에 기록.
- **코퍼스**: 가상 회사 「주식회사 하울」 사내문서 210종(교차 참조 내장). *실제 데이터·개인정보 없음.*

---

## 💡 배운 점 · 한계

- **검색 방식 전환엔 추가 파이프라인이 필요**하다 — 벡터/키워드는 청킹·색인이면 되지만, GraphRAG는 그 위에 엔티티·관계 추출 → 그래프 구축 → 그래프 검색이 더 필요하다.
- **노드 속성(타입·설명·임베딩)이 없으면 문서 연결이 표현되지 않는다** — 그래프 계열의 성패는 추출 품질에 달려 있다.
- 현재 코퍼스는 가상 생성 문서다. GraphRAG 본연의 성능을 최종 판정하려면 커뮤니티 요약 고도화 + 실제 문서로 추가 검증이 필요하다.

---

## 📄 더 보기

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — **코드 지도**(무슨 코드가 어디, X 고치려면 어디)
- [`CLAUDE.md`](CLAUDE.md) — 설계 방향·규칙·전체 결과 정리
- [`docs/WORKLOG.md`](docs/WORKLOG.md) — 작업 타임라인
- [`docs/graph_quality_design.md`](docs/graph_quality_design.md) — 그래프 품질 개선 설계(L1~L5)

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
| **★ HippoRAG2 (공식, OpenIE+PPR)** | **0.806** | 1.00 | 0.92 | 1.00 | 0.25 |

> **hybrid = graphrag_e2b_hybrid = HippoRAG2, judge 0.806 3자 공동 1위** (같은 e5 임베딩 통제).

### 핵심 발견

1. **GraphRAG 저성능은 방법론의 구조적 한계가 아니라 구현 성숙도다** — 자체 그래프(graphrag_e2b 0.333)와 **공식 HippoRAG2 PPR(0.806)** 의 격차가 이를 증명한다. 같은 임베딩(e5)에서, OpenIE+**Personalized PageRank**+passage노드+node specificity를 갖춘 성숙한 구현은 평면 하이브리드와 동률에 도달한다.
2. **올바른 검색층이면 그래프가 평면을 따라잡는다** — 자체 그래프도 **직접 청크 벡터검색을 RRF로 융합**하니 **judge 0.333 → 0.806**(2.4배). 원인 규명 5단계에서 **재랭커 임베딩 버그 + 단일 검색 기법**을 잡아낸 결과.
3. **"하이브리드 > 단일검색"은 도메인 특이적** — 공개벤치(HotpotQA n=100)에선 세 검색이 **통계적 동률**(McNemar 무유의차). 자체 코퍼스(한국어 사내문서)의 우위가 일반법칙은 아님을 **외부검증으로 확인**.
4. **벤치마크 함정 — 결론을 스스로 반증하라** — 약judge·단일검색·미세버그가 결론을 왜곡한다. "그래프=구조적 한계", "HippoRAG=버그로 저하" 두 결론을 **실측으로 뒤집었다**(강judge 교차검증·공개벤치·레버 애블레이션).

> 결론·용도별 선택: [`docs/RESULT.md`](docs/RESULT.md) · 전체 여정: [`CLAUDE.md`](CLAUDE.md) §10.5~10.11.

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
| **graphrag_e2b_l5 / _adaptive / _hybrid** | 커뮤니티 요약 / 라우팅 / 그래프+직접벡터 | ✅ |
| **★ HippoRAG2** | 공식 구현 — OpenIE 지식그래프 + Personalized PageRank | ✅ **최상위 동률** |
| RAPTOR / LightRAG / … | 트리 계층요약 / 듀얼레벨 등 | 🔜 (RAPTOR 착수점 확정) |

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

- **단일 승자는 없다 — 용도별 2 프로필**: 범용은 `hybrid`(벡터+키워드), 조직 특화·연결성·전역 질의는 `graphrag_e2b_hybrid`/HippoRAG2. → [`docs/RESULT.md`](docs/RESULT.md).
- **그래프 저성능 = 방법론이 아니라 구현 성숙도** — 자체 그래프(0.33)와 공식 HippoRAG2 PPR(0.806)의 격차가 증명. 성숙한 OpenIE+PPR+passage노드면 평면과 동률.
- **연결성 ↑ ≠ 성능 ↑ (반직관)** — 과연결 그래프가 핀포인트엔 오히려 노이즈. 그래프의 값어치는 연결성의 양이 아니라 **질문이 실제로 연결을 요구하는지**에 달렸다.
- **코퍼스 품질 실측** — 자체 코퍼스 88%가 자동생성 near-duplicate(TTR 0.156)였음 → 수치에 아티팩트 혼입. 실무 신뢰는 **외부(HotpotQA)·손작성** 위주.
- **방법론이 결과보다 셀링포인트** — 통제·애블레이션·강judge 교차검증·**자기 반증**. 그럴듯한 핑계를 실측으로 걷어내는 게 검증의 값어치.

---

## 📄 더 보기

> **문서 규칙 — 파일을 늘리지 않는다.** 주제별 **5개 문서**로 고정하고, 새 내용은 **해당 문서에 절(節)로 추가**한다(날짜 또는 주제로 구분). 정말 새 축일 때만 문서를 만든다.

| 문서 | 담는 것 |
| --- | --- |
| [`docs/RESULT.md`](docs/RESULT.md) ← **여기부터** | 핵심 압축(한 장 결론) · 용도별 2 프로필 |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | 외부 검증(공개벤치) 설계·실측·문제 진단 |
| [`docs/GUIDE.md`](docs/GUIDE.md) | 실행 가이드 · 코드 지도 · 기술 레퍼런스 · 논문 정리 |
| [`docs/DESIGN.md`](docs/DESIGN.md) | 그래프 품질 개선(L1~L5) · MES·WMS 도메인 확장 |
| [`docs/LOG.md`](docs/LOG.md) | 날짜별 작업 타임라인 · TODO |

- [`CLAUDE.md`](CLAUDE.md) — 설계 방향·규칙·전체 결과(§10.5~10.11)

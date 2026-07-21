# CLAUDE.md

> **RAG 방법론 비교·검증 프로젝트**의 구상·설계 문서. 여러 RAG 기법과 여러 임베딩 방법을 직접 구현·테스트하여 비교하는 것이 목적이다. 아직 코드가 없는 **설계 단계**이며, 개발이 진행되면서 점진적으로 채워나갈 살아있는 문서다. Claude Code는 이 프로젝트에서 작업할 때 이 문서의 방향과 규칙을 따른다.

## 1. 프로젝트 개요
- **풀려는 문제**: 파편적으로 흩어져 있는 문서들에 **연결성을 부여**하고 문서 간 **관계·참조를 강화**하여, 채팅에서 **불필요한 정보가 아니라 정확한 문서 근거**로 답하게 한다. 나아가 특정 프로젝트/프로그램 안에서 **필요한 기능·자료를 정확히 전달**하는 데 활용한다.
- **접근**: 이 "문서 연결성" 문제를 가장 잘 푸는 방식을 찾기 위해, 여러 RAG 기법을 같은 코퍼스·같은 평가셋으로 돌려 **비교·검증(benchmark)** 한다. (관계 중심 문제라 그래프 계열 RAG가 특히 핵심)
- **비교 목표**: 각 RAG 기술이 **문서(특히 문서 간 연결성)** 에서 드러내는 **특징점·강점을 식별·비교**한다 — 어떤 기법이 어떤 문서/질문 유형(단일 사실 vs 다중 홉 vs 관계·인과형)에 강한지를 가린다. (검증표 §6.1)
- **핵심 설계**: 모든 RAG 기법을 **공통 인터페이스** 뒤에 두고, **임베딩 방법**을 교체 가능한 축으로 두어, 동일한 평가 하니스로 성능을 비교한다.
- **현재 상태**: 구현 진행 중. 공통 인터페이스 + 평가 하니스 + **검색 3종(의미/키워드/하이브리드)** 동작. 다음은 그래프 계열(§2). 베이스 스택: Python + LlamaIndex + Gemini.

## 2. 비교 대상 RAG 기법
> ⚠️ 후반부 기법일수록 최신/소수 연구다. 구현 전 **반드시 원논문·공식 레포로 알고리즘을 확인**하고, 아래 설명은 출발점으로만 사용한다. 추측으로 구현하지 말 것.

| 기법 | 한 줄 개념 | 확인 필요 |
| --- | --- | --- |
| **Standard RAG** | 벡터 검색 → top-k 청크 → LLM 생성 (베이스라인) | 낮음 — LlamaIndex 기본 |
| **GraphRAG** | 지식그래프 구축 + 커뮤니티 요약 기반 검색 (Microsoft) | 공식 구현/논문 확인 |
| **LightRAG** | 그래프 기반 경량 RAG, 듀얼레벨(로컬/글로벌) 검색 | 공식 레포 확인 |
| **HippoRAG2** | 지식그래프 + Personalized PageRank 기반(해마 모사) 검색, HippoRAG 개선판 | 논문/레포 확인 |
| **LeanRAG** | 지식그래프 + 계층적/의미 집약 기반 검색(추정) | **세부 확인 필수** |
| **CausalRAG** | 인과 그래프/인과관계를 검색에 반영(추정) | **세부 확인 필수** |
| **HugRAG** | 계층적 인과 지식그래프 + **hierarchical causal gating**(계층적 인과관계 적용 + local spurious node 제거). 관계 중심 벤치마크 **HolisQA** 제안 | arXiv 논문 확인 |

- 각 기법은 동일한 인터페이스(아래 §4)를 구현하는 **플러그형 백엔드**로 추가한다.
- 일부 기법은 LlamaIndex에 없고 **자체 라이브러리/레퍼런스 구현**을 쓴다 → 얇은 어댑터로 공통 인터페이스에 맞춘다.

## 3. 임베딩 방법 (비교 축)
> 여러 임베딩을 교체하며 검색 품질에 미치는 영향을 본다. 후보(미정, 결정 시 추가):

| 후보 | 메모 |
| --- | --- |
| OpenAI `text-embedding-3` (small/large) | 강력한 베이스라인 |
| Voyage AI | Anthropic 추천 임베딩 공급자 |
| 오픈소스 (예: BGE, E5, GTE) | 로컬 실행·비용 0, sentence-transformers |

> **현재 실행 기본값**: 생성·임베딩 모두 **Google AI Studio(Gemini)** — 생성 `gemini-2.5-flash`(무료 티어, 1M 컨텍스트), 임베딩 `gemini-embedding-001`(무료 티어). 키 1개(`GEMINI_API_KEY`)로 동작. (lite는 길이 제한 이슈로 제외) 비용 더 낮추려면 임베딩을 로컬 오픈소스(BGE/E5)로 교체($0). Claude(`claude-opus-4-8`/`claude-sonnet-4-6`)·OpenAI 임베딩 백엔드도 팩토리에 남겨 비교축으로 사용 가능(Anthropic은 임베딩 미제공). 정확한 모델 ID·가격은 각 공급자 문서로 확인.

## 4. 아키텍처 구상
비교 실험이 핵심이므로, **(RAG 기법) × (임베딩) × (평가셋)** 을 격자처럼 돌릴 수 있어야 한다.

```text
                 ┌─────────── 공통 코퍼스 (data/) ───────────┐
                 ▼                                            ▼
        [임베딩 백엔드 N개]                          [공통 평가셋 Q/A]
                 │                                            │
                 ▼                                            ▼
   ┌─────────────────────────────┐                  ┌─────────────────┐
   │ RAG 백엔드 (공통 인터페이스) │  ◄── 동일 질의 ──│  평가 하니스      │
   │  index(corpus) / query(q)   │  ── 답변+출처 ──►│  (지표 집계)      │
   └─────────────────────────────┘                  └─────────────────┘
     ▲   ▲   ▲   ...
   Standard GraphRAG LightRAG ...   ← 각각이 같은 인터페이스 구현
```

**공통 RAG 인터페이스(개념)** — 모든 기법이 이 계약을 따른다:
- `index(documents) -> 영속 인덱스` (구축/저장)
- `query(question) -> { answer, contexts(출처 청크/노드), metadata }`

LlamaIndex는 **Standard RAG의 기본 구현 + 문서 로딩/평가 유틸**로 활용하고, 특수 기법은 어댑터로 감싼다.

### 4.1 문서 연결성 계층 (이 프로젝트의 차별점)
단순 청크-벡터를 넘어, **문서/청크/엔티티를 노드로, 관계·참조를 엣지로** 보는 그래프 표현이 핵심이다. 기법마다 노드·엣지를 만드는 방식이 다르고, 그 차이가 곧 비교 대상이다.

```text
문서 코퍼스 ──► [추출] ──► 노드(문서/청크/엔티티) + 엣지(관계/참조/인과/계층)
                                    │
질문 ──► 검색: 벡터 유사도 + 그래프 탐색(이웃/다중홉/PageRank/인과경로)
                                    │
        컨텍스트 구성: 관련 노드 + 연결 경로 ──► LLM ──► 답변 + 출처 + 연결 근거
```

- **노드/엣지 생성**: 그래프 계열은 LLM 기반 엔티티·관계(또는 인과/계층) 추출로 엣지를 만든다. 이 추출 품질이 연결성의 성패를 좌우한다.
- **검색 = 벡터 + 그래프**: 벡터 유사도로 진입점을 잡고, 그래프 탐색(이웃 확장·다중 홉·PageRank·인과 경로)으로 **흩어진 근거를 연결**한다.
- **답변 시 연결 근거 노출**: 출처 청크뿐 아니라 **참조한 문서 간 관계 경로**를 함께 반환해, "왜 이 문서들이 답인지"를 보인다.
- **공통 인터페이스 확장**: `query()` 반환의 `metadata`에 **사용한 그래프 경로/관계**를 담아, 평가 하니스가 연결성 품질도 측정할 수 있게 한다.

## 5. 디렉토리 구조 (현재 구현)
> src 레이아웃 패키지(`ragbench`). 새 기법은 `methods/` 에 어댑터 추가 + `registry.py` 한 줄.

```text
RAG_project/
  src/ragbench/
    core/
      interface.py    # RagBackend(ABC), QueryResult, RetrievedContext  ✅
      config.py       # YAML/기본값 설정 로딩  ✅
    ingest/loader.py  # 문서 로드 (공유)  ✅
    embeddings/factory.py  # 임베딩 백엔드 (openai → voyage/bge 추가 예정)  ✅
    llms/factory.py        # 생성 LLM (anthropic 기본)  ✅
    methods/standard.py    # Standard RAG (LlamaIndex VectorStoreIndex)  ✅
    registry.py     # 이름 → 기법 매핑  ✅
    cli.py          # index / query / eval 진입점  ✅
    eval/           # 평가 하니스 — dataset·metrics·harness  ✅
  config/default.yaml   # 기법/임베딩/청킹/top-k 설정  ✅
  config/eval_sample.yaml  # 샘플 평가셋(Q/A·정답근거)  ✅
  data/            # 원문 코퍼스 (커밋 제외, 샘플 1개 포함)
  storage/         # 기법별 인덱스 영속화 (커밋 제외)
  results/         # 실험 결과·지표 (다음 단계)
  pyproject.toml   # 패키지·콘솔스크립트(ragbench)·의존성  ✅
  requirements.txt · .env.example · .gitignore  ✅
```

### 5.1 환경 구성 · 실행 (uv)
```bash
# 1) 가상환경 + 설치
uv venv --python 3.10 .venv
uv pip install -e .

# 2) 키 설정 (생성+임베딩 = Google AI Studio)
cp .env.example .env   # GEMINI_API_KEY 채우기 (GOOGLE_API_KEY 도 인식)

# 3) 인덱싱 → 질의 → 평가 (Standard RAG)
.venv/bin/ragbench index --method standard --data data
.venv/bin/ragbench query --method standard "이 프로젝트의 목표는?"
.venv/bin/ragbench eval  --method standard --eval-set config/eval_sample.yaml
```
- 설정 변경은 `config/default.yaml`(공급자·모델·청킹·top-k) 또는 `--config` 로.
- `eval` 은 검색 지표(hit/recall@k/precision@k/MRR) + 지연 + 답변 키워드 적중을 집계해 `results/<method>.json` 에 저장(재현용).
- **검증 상태**: ✅ **end-to-end 통과** — index→query→eval 모두 동작(생성 `gemini-2.5-flash`, 임베딩 `gemini-embedding-001`, 키 `GEMINI_API_KEY`).
- **코퍼스**: `data/company/` — 가상 「주식회사 하울」 사내문서 **17종**, 문서 간 교차 참조 내장. 인물명은 동화 캐릭터, 직급은 평범.
  - 회사규정(연차·휴가·복지) / 행정(문서양식·복지법률·직원현황·경비·장비장부) / 부서별(보고서·사내규정·일정) / 양식(휴가신청서 F-101·주간보고서 F-301) / 통계(연차휴가·인력 직급·직무·부서 교차) / 교육(신입 온보딩 vs 재직 직무교육)
  - 조직: 3개 사업부(농업·인력·신사업개발) + 지원(인사·경영지원).
- **코퍼스 규모 확장 (210종)**: 손작성 규정 24종 + **생성기**([scripts/generate_large_corpus.py](scripts/generate_large_corpus.py))로 186종 추가 — 직원 프로필 33·프로젝트 계획/진행 84·월간보고 30·회의록 30·공지 8·장비확장. **사람↔프로젝트↔장비↔회의↔보고↔부서**가 촘촘히 교차 참조 → 그래프 계열이 의미를 가질 엔티티·관계 확보. (재실행 가능)
  - 효과: 건초더미가 커져 의미검색 recall **하락**(단일 1.0→0.91, 다중 0.79→0.71) — 더 현실적·변별력↑.
- 각 (손작성) 문서는 **문장형 본문**(규정·행정·부서별·양식·통계·교육: 채용·인사평가/승진·급여·정보보안·출장·사업계획·인수인계 포함).
- **현실성 강화**: ① 모든 문서에 **제정일·최종개정일**(신규 규정은 2024 제정) ② 문서 간 인용은 **코드가 아니라 문서명**(「연차 유급휴가 규정」)으로 — 문서번호는 헤더 식별용만 ③ **요일·시간 규칙을 교차로 연결**(예: 금요일 입사자→다음 주 월요일 급여서류 제출, 급여 25일 지급·휴일이면 직전 영업일, 월요일 연차→직전 주 목요일 신청, 경비 25일 18시 마감).
- **평가셋 36문항** — 유형 태그(single 11 / multi 12 / relational 5 / **global 8**). global=전역·종합형(GraphRAG 강점 영역), 정답근거가 넓어 recall@k 대신 keyword_recall·LLM-judge로 채점. `config/eval_sample.yaml`.
- **검색 3종 비교**(210문서, 36문항) — 백엔드: `standard`(의미), `bm25`(키워드), `hybrid`(벡터+BM25 RRF).

  | (recall@4; global은 kw_recall) | 키워드 | 의미 | 하이브리드 |
  | --- | --- | --- | --- |
  | 단일 | 0.818 | 0.909 | 0.909 |
  | 다중 | 0.736 | 0.708 | **0.764** |
  | 관계 | 0.900 | 1.000 | 1.000 |
  | 전역(kw) | 0.531 | 0.802 | **0.812** |

  - **하이브리드가 거의 전 유형 최고/동률.** 전역에서 **키워드 급락(0.53)** — 의미 이해 필요. 다중·전역에 개선 여지 → 그래프 차례.
  - 24문서 시절 깨끗/노이즈 비교(하이브리드가 노이즈에 더 취약 — BM25가 유사어 함정 흡수)는 `results/archive_24doc/`에 보존.
- **GraphRAG 구현·평가** ([methods/graphrag.py](src/ragbench/methods/graphrag.py)) — LlamaIndex `PropertyGraphIndex` + `SimpleLLMPathExtractor`. 구현 중 해결한 이슈: 추출/합성 LLM `max_tokens=8192`(2048이면 MAX_TOKENS), `nest_asyncio.apply()`(중첩 async), `embed_kg_nodes=False`(임베딩 3000/분 한도 회피).

  **4종 비교 결과 (210문서, 36문항, recall은 global 제외)**:

  | 방식 | recall | keyword_recall | latency |
  | --- | --- | --- | --- |
  | bm25 | 0.80 | 0.80 | 3.2s |
  | standard | 0.84 | 0.94 | 3.3s |
  | **hybrid** | **0.86** | 0.92 | 3.4s |
  | graphrag | **0.24** | 0.67 | **10.6s** |

  - ⚠️ **GraphRAG가 크게 패배**(recall 0.24, 3배 느림). **정직한 원인 분석**:
    1. **`embed_kg_nodes=False`로 검색기를 약화**시킴(임베딩 할당량 회피용 타협) → 벡터 랭킹 없이 키워드/엔티티 매칭만 → 관련 청크 다수 누락, 답변이 "확인 불가/참조 바람"으로 회피. **가장 큰 요인이며 GraphRAG의 본래 성능이 아님.**
    2. 우리 평가셋은 **핀포인트 사실·다중 홉** 중심 — GraphRAG의 강점은 **전역 sensemaking**("주요 주제는?"). 그런데 global조차 kw 0.72로 평면(0.81) 못 넘음.
    3. recall@k가 그래프의 무랭킹 대량 검색과 안 맞음.
  - **공정 비교를 위한 TODO**: `embed_kg_nodes=True`(벡터 랭킹 노드 검색)로 **throttle 하여 재인덱싱** 필요. 현재 결과는 "할당량 회피용 약화 config"의 GraphRAG임을 명시.

  **★ 공정 재대결 (3대 원인 모두 해소)** — rich 코퍼스(210, LLM 본문 생성) + `embed_kg_nodes=True` + **로컬 임베딩(multilingual-e5-small, GPU)** 으로 rate limit 제거. 4종 동일 임베딩(local) 비교:

  | 방식 | recall@k | keyword_recall | latency |
  | --- | --- | --- | --- |
  | bm25 | **0.80** | 0.76 | 3.4s |
  | standard | 0.69 | 0.87 | 3.4s |
  | hybrid | 0.78 | 0.84 | 3.0s |
  | graphrag | **0.30** | 0.67 | **10.6s** |

  - **GraphRAG가 공정 config로도 recall 최하·3배 느림.** 단, **recall@k는 그래프 검색에 구조적으로 불리**(연결 노드 다수를 무랭킹 반환 → top-4 출처 매칭에 안 맞음). keyword_recall(답변 품질)은 graphrag도 relational 1.0/multi 0.75로 답변 자체는 무난.
  - **결론**: 이 도메인(연결 사내문서 + 핀포인트 사실 질의)에선 **평면 검색(bm25/hybrid)이 그래프보다 정확·빠름** — 공정 조건에서도. GraphRAG 강점(전역 sensemaking)은 평가셋이 핀포인트 중심이라 안 드러남.
  - **남은 공정성 보강**: recall@k 대신 **LLM-judge(답변 정답성)** 로 재채점해야 그래프 검색 편향 제거. (`eval --judge`)
  - 인프라: 로컬 임베딩 백엔드 추가([embeddings/factory.py](src/ragbench/embeddings/factory.py) `local`), `config/local.yaml`. GraphRAG 이슈 추가 해결: `embed_kg_nodes=True`는 임베딩 rate limit 유발 → 로컬 임베딩으로 회피.
- 평가 도구: `eval --judge`(LLM-as-judge), `compare`(파일명 라벨 비교표). 결과: `results/{standard,bm25,hybrid}_{clean,noise}.json` (6종).
- **노이즈 토글**: `data/noise/`(별도 폴더, 5종 — 무관 문서·유사어 함정·구버전 충돌)을 두어 켜고 끄기 쉽게 함.
  - 깨끗(기본): `ragbench index` → `data/company`만. / 노이즈 포함: `ragbench index --data data` → company+noise(28문서).
  - **노이즈 영향**(측정됨): recall@4는 유지되나 keyword_recall 0.977→0.955·MRR·precision 하락. spurious 노드(`외부_세미나_후기`=타사 100만원, `연차규정_2023_폐지본`=구 수치)가 top-k에 침투 → LeanRAG·HugRAG의 노이즈 제거 강점을 잴 대상 확보. 노이즈 결과 기록: [results/standard_noise.json](results/standard_noise.json).
- **주의**: 생성 모델 `gemini-2.5-flash`는 thinking에도 출력 토큰을 써서 `max_tokens`가 작으면 `MAX_TOKENS`로 조기 종료됨 → `config/default.yaml`에서 2048로 설정.

## 6. 평가 / 검증 (이 프로젝트의 핵심)
- **공정 비교**: 모든 기법이 같은 코퍼스·같은 질문셋·같은 생성 LLM을 쓰고, **임베딩/기법만 변수**로 둔다.
- **확정 지표** (실제 계산·사용 중 — [eval/metrics.py](src/ragbench/eval/metrics.py)·[harness.py](src/ragbench/eval/harness.py)):
  - **답변 품질(주력)**: **`judge_correct`(LLM-as-judge)** ← 아래 §6.0 · 보조 `keyword_recall`
  - 검색 품질(보조): `hit`, `recall@k`, `precision@k`, `MRR` — *그래프 계열엔 구조적 불리(§6.0)라 참고용*
  - 비용/성능: `latency_s`
- **향후 후보(미구현)**: nDCG, faithfulness(근거 충실도), 인용 정확성, 토큰/API 비용, 인덱스 크기.
- **재현성**: 실험 설정과 결과를 `config/`·`results/`에 기록해 동일 조건 재실행 가능하게.

### 6.0 LLM-as-judge — 이 벤치마크의 주력 채점 기준 ★
> 구현 [eval/judge.py](src/ragbench/eval/judge.py) · 실행 `ragbench eval --judge` · 질문당 LLM 1회(비용 有)

- **정의**: 「질문 + 참조 정답(reference_answer) + 모델 답변」을 judge LLM에 주고, **핵심 사실이 일치하면 1, 아니면 0**만 출력하게 해 이진 채점. 평가셋의 `reference_answer`가 정답 기준(36/36 사실 기반).
- **★ 왜 recall@k가 아니라 judge가 주력인가**: recall@k는 검색된 top-k의 **출처 파일명**을 정답 근거와 매칭한다. 그런데 그래프 계열은 ① **무랭킹으로 다량 반환**해 top-k 매칭에 안 맞고 ② **HippoRAG는 원문 문단만 반환(파일명 소실)** 해 recall@k 산출 자체가 불가. → recall@k는 **기법에 따라 구조적으로 불리**하다. judge는 "**최종 답이 맞았나**"만 보므로 검색 방식 편향이 없다. (그래프 기법 평가는 judge·keyword_recall로 한다 — §10.4·§10.10)
- **신뢰성 교차검증(중요)**: 초기 약judge(로컬 gemma)가 **그래프 점수를 부풀렸음**을 발견 → **강judge(Gemini)로 재채점**하니 결론이 뒤집힘(global 0.38→0.00). 이후 전 실험은 **강judge 기준**으로 통일(§10.5).
- **도메인·언어 중립화**: 초기 프롬프트가 *"사내문서 기반 질의응답 채점"* 이라 영어 공개벤치(HotpotQA)에 부적합 → **도메인·언어 중립 문구로 교정**. 교정 후 재채점해도 결론 불변(0.71→0.69 등 노이즈 내) → 언어 confound 없음 확인(§10.7).
- **정직한 한계**: ① **judge↔사람 라벨 상관 미검증** ② 논문 수치와 **메트릭이 다름**(논문 F1 vs 우리 judge) → 절대값 직접 비교 불가(§10.11) ③ n=36~100이라 **±0.05 흔들림**·유의성 부족(§10.6) ④ judge도 LLM이라 생성모델과 같은 계열이면 편향 가능(그래서 강judge 교차검증이 필수).

### 6.1 검증표 (기법별 특징점 × 검증 포인트)
> 각 기법이 **문서 연결성**을 어떻게 모델링하고, 어떤 질문 유형에 강할 것으로 기대되며, 무엇으로 검증할지를 한눈에. 실제 수치는 실험으로 채운다(아래는 가설).

| 기법 | 연결성 모델링 | 기대 특징점·강점 | 핵심 검증 포인트 |
| --- | --- | --- | --- |
| **Standard RAG** | 없음 (독립 청크 벡터) | 단순·빠름, 단일 사실 질의 | 베이스라인 — 단일 질의 정확도/지연/비용 |
| **GraphRAG** | 엔티티-관계 KG + 커뮤니티 요약 | 광범위·종합형(global) 질문, 주제 요약 | global 질문 답변 품질, 커뮤니티 요약 유용성 |
| **LightRAG** | KG + 벡터 이중, 듀얼레벨 | 경량·증분 갱신, 로컬+글로벌 균형 | 인덱싱 비용/속도, 증분 갱신, 검색 적중 |
| **HippoRAG2** | KG + Personalized PageRank | **다중 홉(multi-hop)** 연상, 분산 근거 연결 | multi-hop 정답률, 연결 경로 정확성 *(확인필요)* |
| **LeanRAG** | 의미 집약 + 계층 KG | 중복·노이즈 억제, 계층적 요약 검색 | 노이즈 감소율, 계층 검색 효과 *(확인필요)* |
| **CausalRAG** | 인과 그래프 | 원인-결과/이유형 질문, 환각 감소 | 인과형 질문 정확도, faithfulness *(확인필요)* |
| **HugRAG** | 계층적 인과 KG + causal gating | 관계 중심 질문, spurious 노드 제거 | HolisQA류 관계 질문 성능, 노이즈 제거 효과 *(확인필요)* |

> **연결성 전용 지표(검토)**: 관계 경로 정답성, 다중 홉 근거 도달률, 불필요 노드(spurious) 비율 — 단순 recall@k로는 안 잡히는 "연결성 품질"을 측정하기 위함.

## 7. 개발 규칙
- **공통 인터페이스 우선**: 새 RAG 기법은 반드시 §4의 인터페이스를 구현하는 어댑터로 추가. 평가 하니스는 기법을 몰라야 한다.
- **구현 전 출처 확인**: 각 기법은 원논문/공식 레포를 먼저 확인하고 구현. **알고리즘을 추측으로 작성하지 않는다.**
- **설정 분리**: 기법·임베딩·모델·청킹·top-k 등은 설정 파일/환경변수로. 하드코딩 금지.
- **비밀키 관리**: API 키는 `.env`, `.gitignore`에 `.env`·`data/`·`storage/` 추가. 키 커밋 금지.
- **점진적 추가**: 먼저 Standard RAG로 평가 하니스를 끝까지 검증한 뒤, 기법을 하나씩 붙인다.
- **단순함 우선**: 비교에 필요 없는 추상화는 만들지 않는다.

## 8. 열린 결정 (Open Questions)
- 비교할 임베딩 방법 최종 목록
- 평가 코퍼스·질문셋(정답 포함) 출처 (공개 벤치마크 vs 자체 구축)
- 평가 지표 확정 및 자동 채점 방식(LLM-as-judge 사용 여부)
- 그래프 기반 기법용 그래프 저장소(예: networkx/Neo4j) 선택
- 기법별 논문·공식 레포 **출처 확보 완료** (부록 A) → 남은 일은 구현 시 각 논문의 세부 알고리즘 정독
- **HolisQA**(HugRAG 논문 제안, 관계 중심 QA) 평가셋 채택 검토 — 공개 형태·라이선스 확인 필요, 그래프 계열 비교에 적합
- 결과 시각화/리포트 형식

## 9. 다음 단계
1. ~~`core` 공통 인터페이스 + `eval` 하니스 + **Standard RAG** 끝단 동작 검증~~ ✅ 완료
2. **평가 코퍼스·질문셋 확장** — 샘플 1개 → 다문서 코퍼스 + 다홉/관계형 문항(지금은 지표가 자명) (§6, §8 해소)
3. 임베딩 백엔드 2~3개 연결해 임베딩 축 비교 가능하게
4. 그래프 계열(GraphRAG/LightRAG) → HippoRAG2 → 나머지 순으로 기법 추가 (각 구현 전 출처 확인)
5. 전체 격자 실험 실행 + 결과 리포트

## 10. 배운 점 · 한계점

### 10.1 배운 점
1. **검색 방식 전환에는 추가 처리가 필요하다.** 의미(벡터)·키워드(BM25) 검색은 문서를 청킹·색인하면 바로 동작하지만, GraphRAG를 활용하려면 그 위에 **① LLM 엔티티·관계 추출 → ② 그래프 구축 → ③ 그래프 기반 검색**이라는 별도 파이프라인이 더 필요하다.
2. **노드 메타데이터(관계·속성)가 없으면 문서 간 연결이 어렵다.** 노드 간 관계, 노드 속성(엔티티 type·description), 엔티티 정보가 빈약하면 그래프가 문서 사이의 연결을 표현하지 못한다. 실제 본 프로젝트의 그래프는 엔티티에 **이름만 있고 type·description·임베딩이 모두 비어 있어**(엔티티 4907개 전부 임베딩 None) 문서 연결을 거의 담지 못했다.

**결과 (4종 공정 비교 — 로컬 임베딩, rich 코퍼스 210문서, 36문항)**

| 방식 | recall@k | keyword_recall | latency |
| --- | --- | --- | --- |
| bm25 (키워드) | 0.80 | 0.76 | 3.4s |
| standard (의미) | 0.69 | 0.87 | 3.4s |
| hybrid (의미+키워드) | 0.78 | 0.84 | 3.0s |
| **graphrag** | **0.30** | 0.67 | **10.6s** |

위 표에서 GraphRAG가 낮게 나온 것은 **방법론의 한계가 아니라, 노드와 속성이 비어 있어 문서 관계가 표현되지 않은 데서 비롯된 문제**다. (`SimpleLLMPathExtractor`의 평면 트리플 + 엔티티 임베딩 누락 + 커뮤니티 요약 미구현)

### 10.2 한계점
- 본 코퍼스는 **가상으로 생성된 문서**이므로, GraphRAG의 정확한 성능을 판정하려면 **커뮤니티 요약 등 본연의 구성**과 **풍부한 실제 문서 자료**로 추가 검증이 필요하다. 현재 수치는 "빈약한 그래프 구성"에서의 결과이며, 방법론에 대한 최종 평가가 아니다.
- **★ 이상적 상황(상한값) 전제 — 반드시 명시.** 이 벤치마크는 **가상 코퍼스 + 깨끗한 「문서명」 상호참조**에서 수행됐다(연결성 테스트를 위해 문서가 서로를 명시적·결정론적으로 인용하도록 설계). 따라서:
  - **그래프의 연결성 이득·L4(문서참조 그래프) 효과는 낙관적(best-case)** — 깨끗한 인용이 만든 최선값. 실제 문서는 참조가 지저분(암시적·표현 다양·부분 인용)해 **L4·연결성 이득에 갭(하락)** 이 생긴다. L4 수치는 "상한선"으로 읽어야 한다.
  - 절대 정답률 수치도 이상적 코퍼스 기준. 실무 적용 전 **실제(지저분한) 문서 재검증 필수**.
  - **단, 상대적·메커니즘 결론은 견고**: LLM-judge는 문서명이 아니라 **사실**을 채점(36/36 사실 기반 reference_answer — "근로기준법 제60조" 등)하므로 "이름매칭" 편향이 없고, "검색층 하이브리드 > 단일 검색"은 검색 메커니즘 결론이라 코퍼스 성격과 무관하다. (recall@k만 문서명 기반이나 그래프에 구조적 불리해 judge로 대체 중)

### 10.3 개선 설계 (구상)
- 그래프 품질 개선 기법 조사·설계는 [docs/DESIGN.md](docs/DESIGN.md)에 정리. 핵심: **SchemaLLMPathExtractor(도메인 스키마)** + **「문서명」 인용 파싱(결정론적 문서참조 그래프)** + 엔티티 임베딩 수정 + (이후) 커뮤니티 요약. 대안: LightRAG·HippoRAG2.

### 10.4 개선 실행 결과 — E2B 산문추출 + 정규화 (2026-07-01)
> **원인은 문서가 아니라 자동 추출 알고리즘이었다**는 §10.1 진단을 직접 검증. 같은 코퍼스를 추출기만 바꿔 재구축.

- **방법**: 작은 로컬 모델(`gemma4:e2b`, M2 Ollama)의 **산문 생성 강점**을 이용 — `이름 \| 유형 \| 설명` 형식으로 뽑아 결정론적 파싱([methods/graphrag_e2b.py](src/ragbench/methods/graphrag_e2b.py)). `SimpleLLMPathExtractor`(평면 트리플)와 달리 **타입+설명**을 채운다. 임베딩은 로컬 e5-small(RTX). **전 과정 로컬·무료**(API 비용 0).
- **정규화 내장**: 영어 `entity` 폴백·관계어 오분류(소속→직원)·관계 표면형(관련/관련됨, 따름/따른다) → 결정론적 규칙으로 통합(`normalize_type`/`normalize_rel`).
- **그래프 품질**(210문서): 엔티티 1,152 · 관계 1,376 · **추출 엔티티 711개 전부 description 보유**(빈약 그래프는 0%). 타입 11종(직원 96·장비 88·프로젝트 76·부서 74·양식 74·규정 43…). 사람↔프로젝트↔부서↔규정↔양식이 실제 관계로 연결.

**4종 공정 평가 (210문서·36문항·LLM-judge, 전부 로컬 e5+gemma)**

| 기법 | recall@k | kw_recall | **judge(정답률)** | latency |
| --- | --- | --- | --- | --- |
| bm25 | 0.80 | 0.73 | 0.61 | 44s |
| standard | 0.69 | 0.85 | 0.72 | 45s |
| **hybrid** | 0.78 | **0.86** | **0.72** | 44s |
| graphrag_e2b | **0.13** | 0.56 | 0.53 | **109s** |

**유형별 judge_correct** — ★ global에서 그래프 역전:

| 기법 | single | multi | relational | **global** |
| --- | --- | --- | --- | --- |
| bm25 | 0.82 | 0.58 | 1.0 | 0.13 |
| standard | 0.82 | 0.75 | 1.0 | 0.38 |
| hybrid | 1.0 | 0.67 | 1.0 | 0.25 |
| **graphrag_e2b** | 0.55 | 0.50 | 0.40 | **0.63** |

- **결론(가설 검증)**: **핀포인트(single/multi/relational)=평면검색 압승, 종합형(global)=그래프 승.** graphrag_e2b가 global judge 0.63으로 1위(bm25의 5배, hybrid의 2.5배)·global kw_recall 0.65로 최고 — **GraphRAG 논문의 "global sensemaking 강점"이 우리 데이터에서 처음으로 실측됨.**
- **recall@k는 그래프에 구조적 불리** 재확인(무랭킹 대량 반환 vs top-4 매칭) → 그래프는 **judge로 평가해야 공정**. 빈약 그래프(recall 0.30)보다 recall은 더 낮아 보여도 judge·global 품질은 개선됨.
- 결과 보존: `results/local_eval/{bm25,standard,hybrid,graphrag_e2b}_judge.json`.
- **남은 개선**: graphrag_e2b의 핀포인트 약점 → 그래프+벡터 하이브리드 검색, 커뮤니티 요약(전역 강화), 관계 라벨 추가 정규화. 다음 기법(LightRAG/HippoRAG2)과 비교.

### 10.5 최종 결론 — "GraphRAG는 왜 지는가"의 진짜 원인 (2026-07-02)
> §10.4까지의 "그래프가 구조적으로 불리"라는 잠정 결론을 **끝까지 검증해 뒤집은** 기록. 상세 타임라인은 [docs/LOG.md](docs/LOG.md) 2026-07-02 참고.

- **검증 절차(5단계)**: ① 약judge(gemma)가 그래프 점수를 부풀림을 **강judge(Gemini) 재채점**으로 확인(global 0.38→0.00) → ② **강추출(Gemini)** 로 그래프 재구축(desc 62%→98%)해도 여전히 낮음 → "구조적 한계"로 오결론 → ③ **재랭커 임베딩 버그** 발견: 검색 노드가 `트리플+원문`인데 e5(512토큰)가 앞의 트리플만 임베딩 → 정답 청크 탈락(수정: 재순위 시 트리플 제거) → ④ "버그가 전부 아님, 검색이 그래프 단일 기법이라 낮다" 통찰 → ⑤ **검색층 하이브리드**(그래프+직접 청크벡터 RRF, `GraphRAGE2BHybrid`).

**최종 비교 (Gemini 그래프·생성·judge)**

| 방식 | judge | single | multi | relat | global |
| --- | --- | --- | --- | --- | --- |
| standard (평면) | 0.833 | 1.00 | 0.92 | 1.00 | 0.38 |
| graphrag_e2b (그래프만) | 0.333 | 0.36 | 0.50 | 0.20 | 0.12 |
| **graphrag_e2b_hybrid (그래프+직접벡터)** | **0.806** | 1.00 | 0.92 | 0.80 | 0.38 |

- **결론**: **GraphRAG 저성능 = 방법론의 구조적 한계가 아니라 ① 재랭커 임베딩 버그 + ② 단일 검색 기법.** 올바른 검색층(그래프+벡터 하이브리드 + 정상 재랭커)이면 **평면 정밀도를 따라잡으며 전역·연결성을 더한다**(0.333→0.806). §10.1~10.4의 "빈약 그래프/구조적 한계" 서사는 이 단계에서 최종 수정됨.
- **교훈**: 벤치마크에서 **약한 judge·단일 검색·미세 버그**가 결론을 통째로 왜곡할 수 있다 → 강judge 교차검증·다중 검색 융합·실제 컨텍스트 검사(무엇이 LLM에 전달되나)를 항상 확인할 것.

### 10.6 방법론의 위치 · 도메인/벤치마크 한계 (2026-07-04)
> "우리 검증이 논문식인가?" 논의 정리 — 과장·과소 없이 위치를 못 박음.

- **방법론(사고 구조)은 논문식이 맞다** — ① 통제 비교(코퍼스·임베딩·생성·청크 고정, 기법만 변수) ② 애블레이션(재랭커/하이브리드/L5/적응형 요소별 기여 분리) ③ **LLM-judge 신뢰성 교차검증**(약judge gemma→강judge Gemini) ④ 가설→**반증**→재검증(“구조적 한계” 결론을 스스로 뒤집음) ⑤ 재현성(config·results·gemma/Gemini 양쪽) ⑥ 한계 명시.
- **논문 "수준"엔 못 미친다** — 자체 **가상 소규모 코퍼스**(36문항 → 유형별 n=5~12, **통계 유의성 없음**·수치 ±0.05 흔들림), 단일 시드(CI 없음), **공식 구현**(microsoft/graphrag 커뮤니티 리포트 등) 미사용, **공개 SOTA 수치 대조 없음**, judge↔사람 라벨 상관 미검증.
- **★ 근본은 도메인/언어 미스매치** — 공개 벤치마크(HotpotQA·MuSiQue·2Wiki·NQ·BEIR)는 **다운로드 가능**하다(못 얻는 게 아님). 다만 **영어·위키피디아 도메인**이라 우리 목표(**한국어·사내문서·문서 연결성**)와 안 맞는다. 한국어 연결성 벤치는 드묾(KorQuAD=독해). **→ 그래서 자체 가상 코퍼스를 만든 것**(§10.2 “이상적 상한값” 한계와 직결).
- **선택지**: A) 공개 영어벤치(HotpotQA 등) **소규모 추가** → 표준 벤치 검증(코드 그대로, 코퍼스만 교체 / 단 우리 도메인은 벗어남). B) 가상 코퍼스 유지 + **상한값 명시**(정직·포트폴리오엔 충분). C) 한국어 연결성 벤치 탐색/구축(비용↑).
- **결론(포트폴리오 프레이밍)**: 셀링포인트는 **결과가 아니라 “검증 과정의 엄밀함”**(반증가능성·통제·judge 교차검증). "SOTA 이김"·"일반화" 과장 금지. 논문급으로 한 발 더 가려면 **A(공개벤치)** 부터.

### 10.7 전 기법 매트릭스 + 공개벤치(HotpotQA) 외부검증 (2026-07-12)
> §10.6의 선택지 **A(공개벤치)** 실행 + 회사 코퍼스 **전 10기법** 매트릭스 완성. judge 프롬프트를 도메인·언어 중립으로 교정([eval/judge.py](src/ragbench/eval/judge.py) — "사내문서" 문구 제거)해 양 코퍼스 공정 채점.

**(1) HotpotQA 공개벤치 — 평면 3종 (n=100, 다중홉 hard, 영어 위키)**
> `config/eval_hotpot.yaml`(100문항) + `data/hotpot/`(고유문단 991). 임베딩 다국어 e5, 생성 LlamaIndex 영어 기본, judge 중립. 결과 `results/local_eval/{bm25,standard,hybrid}_hotpot.json`.

| 방법 | recall | kw_recall | **judge(중립)** |
| --- | --- | --- | --- |
| standard(의미) | 0.81 | 0.75 | 0.69 |
| hybrid | 0.80 | 0.73 | 0.69 |
| bm25(키워드) | 0.75 | 0.68 | 0.64 |

- **McNemar 쌍대검정 — 세 방법 전부 유의차 없음**(standard-hybrid p=0.55, standard-bm25 p=0.13, hybrid-bm25 p=0.33, n=100).
- **★ "하이브리드 > 단일검색"은 도메인 특이적 — 보편법칙 아님.** 회사 코퍼스(한국어·정확한 코드/용어)에선 하이브리드가 이겼으나 HotpotQA(위키·distractor 다수)에선 의미검색만으로 충분하고 BM25가 이득 없음 → 세 방법 동률. §10.6 "도메인 미스매치"를 **실측 확증**.
- **부수 검증**: 외부 표준벤치에서도 하니스가 합리적 수치(hard 다중홉 0.64~0.69) → 코드가 자체 코퍼스에 과적합 안 됨. 언어 confound(judge 한글) 제거 후에도 결론 불변(0.71→0.69 등 노이즈 내).

**(2) 회사 코퍼스 전 기법 매트릭스 (n=36, Gemini graph·생성·judge, 로컬 e5)**
> 재추출 없이 기존 인덱스 재사용. `results/local_eval/{method}_matrix.json`.

| 기법 | judge | recall | single | multi | relat | **global** |
| --- | --- | --- | --- | --- | --- | --- |
| **hybrid**(의미+키워드) | **0.806** | 0.863 | 1.00 | 0.92 | 1.00 | 0.25 |
| **e2b_hybrid**(그래프+벡터) | **0.806** | 0.833 | 1.00 | 0.92 | 1.00 | 0.25 |
| standard(의미) | 0.778 | 0.863 | 1.00 | 0.92 | 0.80 | 0.25 |
| bm25(키워드) | 0.694 | 0.708 | 0.82 | 0.83 | 1.00 | 0.13 |
| graphrag_dynamic | 0.583 | 0.458 | 0.55 | 0.75 | 0.60 | **0.375** |
| e2b(그래프만) | 0.389 | 0.232 | 0.36 | 0.42 | 0.40 | **0.375** |
| e2b_l5(+커뮤니티) | 0.389 | 0.232 | 0.46 | 0.50 | 0.20 | 0.25 |
| e2b_adaptive(라우팅) | 0.389 | 0.179 | 0.36 | 0.50 | 0.20 | **0.375** |
| graphrag(평면트리플) | 0.222 | 0.196 | 0.27 | 0.08 | 0.80 | 0.00 |
| graphrag_schema | 0.083 | 0.196 | 0.00 | 0.08 | 0.40 | 0.00 |

- **★ hybrid = e2b_hybrid 공동 1위(0.806).** 검색층 하이브리드(그래프+직접벡터 RRF)가 평면 하이브리드와 **완전 동률** → §10.5 최종결론 재확인: **그래프 저성능 = 구조적 한계 아니라 "단일 검색 + 재랭커 버그".** 벡터를 되살리면 그래프도 정상급 도달.
- **순수 그래프 검색은 핀포인트에 약함** — 무랭킹 대량 반환이 single/multi/relational에 안 맞음. schema(0.083) 최악.
- **global에선 그래프 계열 역전** — dynamic/e2b/adaptive 0.375 > 평면 0.25. 그래프 sensemaking 강점이 여기서만(단 global n=8, 전 기법 <0.4로 약함).
- **hybrid vs standard(0.806 vs 0.778)는 노이즈 범위**(HotpotQA McNemar가 뒷받침) — "하이브리드 ≥ 단일, 결코 열세 아님"이 견고한 결론.
- **최종 종합**: 이 도메인(연결 사내문서 + 대부분 핀포인트 질의)에선 **하이브리드(평면 또는 그래프+벡터)가 최선**. 순수 그래프는 global 전용. 단 §10.2 상한값·§10.6 통계력 한계 유효.

**(3) HotpotQA 그래프 계열 외부검증 (n=100, 다중홉)**
> §(1)의 평면에 이어 **그래프 계열도 공개벤치로** 검증. **내부 실험과 동일 조건**(추출 gemini-2.5-flash·thinking off, E2B 산문추출·파서, 로컬 e5, 그래프+재랭커 검색, 중립 judge) — 딱 하나 도메인 적응: 엔티티 유형·프롬프트 언어를 **한국어 사내→영어 범용**(`extract_lang: en`, 파서 마커 `[엔티티]`/`[관계]` 유지). 영어 그래프: 엔티티 9,759·desc 79%. 결과 `results/local_eval/graphrag_e2b*_hotpot.json`.
> ⚠️ **핸디캡**: 프롬프트 마커가 한국어라 Gemini가 **설명(desc)을 일부 한국어로 생성**(이름은 정상 영어) → 그래프에 불리한 보수적 조건.

| 기법 | 외부 judge | 외부 recall | 내부 judge | Δ(외−내) |
| --- | --- | --- | --- | --- |
| 평면·의미(standard) | 0.69 | 0.81 | 0.78 | −0.09 |
| 평면·키워드(bm25) | 0.64 | 0.75 | 0.69 | −0.05 |
| 평면·하이브리드 | 0.69 | 0.80 | 0.81 | −0.12 |
| **그래프만(e2b)** | 0.61 | 0.68 | 0.39 | **+0.22** |
| **그래프+커뮤니티(l5)** | 0.65 | 0.67 | 0.39 | **+0.26** |
| **그래프+라우팅(adaptive)** | 0.67 | 0.64 | 0.39 | **+0.28** |
| 그래프+직접벡터(e2b_hybrid) | 0.60 | 0.75 | 0.81 | −0.21 |

- **★ 그래프는 구조적으로 약한 게 아니다 — 외부 다중홉이 확증.** 순수 그래프가 내부 0.39 → 외부 **0.61~0.67**로 점프(Δ +0.22~0.28), 평면(0.64~0.69)에 **통계적 동급**(최고차 2점, n=100). HotpotQA는 설계상 다중홉(엔티티 브리징)이라 **엔티티-관계 탐색이 천연 적합**. → §10.5 "그래프 저성능=핀포인트 도메인 탓, 방법론 아님"을 **남의 데이터로 재확인**.
- **하이브리드 우위도 도메인 특이적 재확인.** 외부에선 **아무도 압도 못 함**(전 기법 0.60~0.69 좁은 밴드, 평면 McNemar 무유의차와 일치). 내부 1위 e2b_hybrid가 외부선 0.60 하위 → 융합 이득이 distractor 많은 위키에선 안 통함.
- **핸디캡(한국어 desc)에도 그래프가 이 수치** → 깨끗한 영어 그래프면 더 높을 여지(보수적 하한).
- **한 줄**: 외부검증이 내부 서사를 뒤집지 않고 **강화** — "그래프=핀포인트 약함/다중홉 강함, 하이브리드 우위=도메인 특이적"이 두 도메인에서 일관. (내부-외부 격차 원인 분석 → §10.8)

### 10.8 내부-외부 격차 원인 분석 — "연결성 문제인가?" (2026-07-12)
> §10.7 그래프 외부검증에서 그래프 계열이 내부 0.39 → 외부 0.61~0.67로 Δ +0.22~0.28 급등. **원인을 단계적으로 실측.** 결론: **연결성은 원인이 아니다(오히려 역상관). 질문-그래프 정렬 + 답변 스타일이 진짜 원인.**

**Step 1 — 질문유형만으로 설명 안 됨.** 내부 e2b 유형별: single .364·**multi .417**·relat .40·global .375. 외부(전부 다중홉) e2b **.610**. 내부 multi(.417)조차 외부(.61)보다 크게 낮음 → "외부가 다 다중홉이라 유리"만으론 잔차 설명 불가.

**Step 2 — 그래프 연결성 실측(역설적 결과).** 두 `graphrag_e2b` 그래프 구조:

| 지표 | 회사(내부) | HotpotQA(외부) |
| --- | --- | --- |
| 엔티티 수 | 1,303 | 8,767 |
| 엔티티간 엣지 | 2,383 | 10,658 |
| 엔티티당 평균차수 | **3.66** | 2.43 |
| 고립 엔티티 % | 0.6% | 2.8% |
| 연결요소 수 | 20 | 511 |
| 최대요소 % | **96.5%**(헤어볼) | 81.9% |

- **회사 그래프가 더 촘촘**(차수 3.66, 96.5% 한 덩어리)한데 그래프 성능은 **더 나빴다**. HotpotQA는 더 파편적(511조각)인데 **더 좋았다**. → **"연결성 좋을수록 그래프 잘한다" 가설 기각(역상관).** "동일 연결성 아니었고, 연결성은 격차 원인도 아님."

**Step 3 — 답변 스타일 + 질문 정렬이 진짜 원인.**

| | HotpotQA 외부 | 회사 내부 |
| --- | --- | --- |
| 정답 단어수(중앙값) | 2 | 6 |
| 짧은답(≤3단어) | 88% | 25% |
| 질문 성격 | 전부 2홉 엔티티 브리지 | 대부분 핀포인트/절차형 |

**★ 가장 깔끔한 지표 — 그래프↔평면 격차 붕괴**: 내부 0.806−0.389=**0.42** → 외부 0.69−0.67=**0.02**. 도메인 난이도(평면도 함께 하락) 상쇄하고도 격차가 0.42→0.02로 붕괴.

**격차 원인 = 3요인 중첩(단일 변수 아님, 정직히):**
1. **질문-그래프 정렬**(주 원인): HotpotQA는 질문이 정확히 2홉 브리지를 요구 → 그래프 탐색의 본업. 회사는 대부분 핀포인트 → 불필요한 이웃을 끌어와 노이즈.
2. **회사 그래프의 과연결**(증폭): 「문서명」 촘촘한 상호참조 → 96.5% 한 헤어볼 → 핀포인트 탐색이 헤맴. **연결성이 많을수록 핀포인트엔 오히려 독.**
3. **답변 findability**: 외부 88%가 ≤3단어 → 그래프가 문단만 잡으면 답 추출 쉬움. 회사 절차형은 종합 필요.

**교훈**: 그래프의 값어치는 **연결성의 양이 아니라 질문이 실제로 연결을 요구하는지**에 달렸다. 사용자가 의도한 "촘촘한 연결성"(§4.1 차별점)은 **핀포인트 질의엔 역효과**였다. 단 내부-외부 Δ는 유형·답변·토폴로지가 뒤섞인 confound라 **단일 원인으로 귀속 불가** — 통제 실험(같은 그래프에서 질문유형만 바꾸기)이 다음 과제.

### 10.9 코퍼스 품질 실측 — 내부 벤치 타당성 재점검 (2026-07-12)
> §10.8 후속. "내부가 낮은 게 ① 도메인 정형·반복 ② 자동생성 저품질 ③ 영문 벤치라서?" 3가설을 **임베딩 상호유사도 + 어휘다양성(TTR)** 으로 실측. 임베딩 로컬 e5, 회사 210 전체 + HotpotQA 210 표본(seed 42).

| 코퍼스 | 평균 쌍유사도 | 최근접 유사(중앙) | 어휘다양성 TTR |
| --- | --- | --- | --- |
| 회사 전체 (210) | 0.891 | **0.969** | 0.163 |
| 회사·생성 (185) | 0.894 | 0.972 | **0.156** |
| 회사·손작성 (25) | 0.907 | 0.952 | **0.379** |
| HotpotQA (210) | 0.747 | 0.830 | 0.345 |

- **H1 도메인 정형·반복 — ✅ 확인.** 회사 상호유사도 0.89·최근접 **0.97**(거의 중복본) vs HotpotQA 0.75/0.83. 회사 문서는 서로 의미가 겹쳐 검색이 구별 못 함 → 전 기법 불리, **§10.8 과연결 헤어볼의 근본 원인**.
- **H2 자동생성 저품질 — ✅ 확인(최강 신호).** 생성 185개 TTR **0.156** = 손작성 0.379의 **절반↓**. **코퍼스 88%가 템플릿 near-duplicate**(월간보고·프로필·회의록 스크립트 생성). → §10.8 "과연결"은 실은 **템플릿 자동생성의 하류 결과**(같은 문구·엔티티 반복 → 밀도만 높고 의미 없는 연결). *(한·영 TTR 절대비교는 토큰화 차 있어 조심 — 한국어 내부 비교 0.156 vs 0.379가 확실한 근거.)*
- **H3 영문 벤치 자체 — ❌ 기각.** 영어가 쉬웠다면 평면도 올라야 하나 **평면은 하락**(standard 0.78→0.69), 그래프만 상승. **손작성 한국어 TTR(0.379) > 영어(0.345)** → 언어가 아니라 문서의 진짜성·다양성이 관건.
- **★ 함의(정직).** 내부 벤치 타당성이 **부분적으로 흔들림** — §10.2 "이상적 상한값"을 넘어 **코퍼스 88%가 저품질 자동생성 near-duplicate**라는 하드넘버 확보. 내부 수치는 방법 차이 + **코퍼스 아티팩트**가 섞임. 단 손작성 25개는 고품질(TTR 0.379>HotpotQA)이라 균일하게 나쁘진 않음. **→ 실무 결론은 외부(HotpotQA)·손작성 위주로 신뢰**, 자동생성 대량분은 참고용. 후속: 생성 코퍼스 품질 개선(다양화·중복 제거) 또는 공개벤치 비중↑.

### 10.10 HippoRAG2 통합·평가 — 성숙한 그래프RAG는 최상위 (2026-07-12)
> §2 로드맵의 다음 기법. 공식 `hipporag`(OpenIE 지식그래프 + **Personalized PageRank**, arXiv 2502.14802)을 얇은 어댑터([methods/hipporag.py](src/ragbench/methods/hipporag.py))로 통합. §7 준수: PPR·OpenIE 재구현 없이 공식 구현 사용.

**검증 충실(임베딩 통제)**: 다른 기법이 전부 로컬 e5라, HippoRAG도 e5로 통일 — e5를 **OpenAI호환 서버**([scripts/e5_openai_server.py](scripts/e5_openai_server.py))로 감싸(다른 기법과 **완전 동일 벡터**) 연결. 생성·judge도 Gemini 동일 → **PPR 검색만 변수**. LLM은 Gemini OpenAI호환 엔드포인트(seed 필드 런타임 패치).

| 기법 | judge | kw_recall | single | multi | relat | global |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.806 | 0.896 | 1.00 | 0.92 | 1.00 | 0.25 |
| e2b_hybrid | 0.806 | 0.859 | 1.00 | 0.92 | 1.00 | 0.25 |
| **★HippoRAG2(PPR)** | **0.806** | **0.907** | 1.00 | 0.92 | 1.00 | 0.25 |
| standard | 0.778 | 0.882 | 1.00 | 0.92 | 0.80 | 0.25 |
| e2b(그래프만) | 0.389 | 0.449 | 0.36 | 0.42 | 0.40 | 0.375 |
| graphrag(평면트리플) | 0.222 | 0.257 | 0.27 | 0.08 | 0.80 | 0.00 |

- **★ hybrid = e2b_hybrid = HippoRAG2 judge 0.806 3자 공동 1위**, HippoRAG2가 kw_recall 0.907로 전체 최고.
- **성숙한 공식 그래프RAG는 최상위 도달** — 같은 e5로 PPR이 평면 하이브리드와 완전 동률, 전 유형 동일. 반면 자체 그래프(e2b 0.389·graphrag 0.222)는 훨씬 낮음 → **§10.5 최종 확증: 그래프 저성능=방법론 아니라 구현 성숙도**(OpenIE+PPR+passage노드 대 SimpleLLMPathExtractor+무랭킹).
- **PPR 다중홉 강점**(multi 0.917 최상위군) — 임베딩 통제했으니 순수 PPR 메커니즘의 값어치. 단 **global 0.25**로 평면과 같음(우리 global 질문이 전 기법 약함) → HippoRAG2는 "강한 핀포인트·다중홉 검색기"로 작동.
- **주의**: `recall@k=0`은 검색 실패 아님 — HippoRAG 반환이 원문 문단(파일명 소실)이라 recall@k 미산출, **judge·kw_recall로 평가**(그래프 기법 방침과 동일). 의존성: hipporag가 torch 2.10→2.5.1·transformers 다운그레이드(+vllm)했으나 기존 파이프라인 정상 확인. `pip install hipporag` 별도(핵심 requirements 미포함, 어댑터 지연 import).
- 인프라: config `hipporag.yaml`(contriever·동작확인용)·`hipporag_e5.yaml`(e5·공정결론용). 결과 `results/local_eval/hipporag_matrix.json`.

### 10.11 외부검증 완결 — HippoRAG2 공개벤치 + 독립 논문 대조 (2026-07-12)
> §10.10 후속. HippoRAG2를 ① 공개벤치(HotpotQA)로 외부검증하고 ② 독립 VLDB 논문의 대규모 통제실험과 대조.

**(1) HippoRAG2 외부검증 (HotpotQA n=100, e5, PPR)** — `results/local_eval/hipporag_hotpot.json`
- judge **0.640** — 외부 클러스터(평면 0.64~0.69)에 안착, 아무도 압도 못 함(§10.7과 일치).
- **★ 07-13 교정 — 0.640은 저하값이 아니라 진짜 수치.** 처음엔 "25/100 recognition-memory JSON 파싱 실패(Gemini shim ↔ HippoRAG 엄격 JSON) → 보수적 하한"으로 봤으나, **두 가설 모두 실측 반증**:
  - **가설 A(버그)**: 파서에 JSON-repair(잘린 출력에서 완전한 트리플만 정규식 복구) 추가 → 파싱에러 25→**0**, judge **0.640→0.640**(불변). 필터 실패 시 graceful 폴백이 복구본과 동등해 점수를 안 깎았음.
  - **가설 B(임베딩)**: e5-small(384d)→**NIM nv-embed-v1(4096d)** 재인덱싱(OpenIE는 Gemini 재사용) → judge **0.640→0.650**(Δ+0.010, 노이즈). 시노님 엣지 폭증(→**151,873**, 20배+)에도 불변 → §10.8 "연결성↑≠성능↑" 재확인.
  - **→ 0.640~0.650이 이 셋업(Gemini 생성·judge)의 HippoRAG2 진짜 성능.** 외부 클러스터(평면 0.64~0.69)에 안착, 아무도 압도 못 함(§10.7 일치). 논문 75.5와의 격차는 고칠 레버가 아니라 **메트릭(F1 vs LLM-judge) + NV-Embed-v2(무료엔 v1만) + 다른 하니스**. 결과 `results/local_eval/hipporag_hotpot{,_repaired,_nvembed}.json`. 인프라: 어댑터 JSON-repair 패치 + 내부LLM/임베딩 override(config `hipporag_llm_*`, NIM 배선).

**(2) 독립 논문 대조 — [In-depth Analysis of Graph-based RAG (VLDB 2025)](https://arxiv.org/abs/2503.04338)** (CUHK+Huawei, 12기법×11데이터셋×GPT-4o judge)
- **프레임워크 수렴**: 논문의 4단계(graph building→index→operator config→retrieval&generation)가 우리 `RagBackend`+하니스 분해와 동일 철학. "기법=연산자 조합" = 우리 §4 플러그형 백엔드.
- **분류 일치**: Table 1이 HippoRAG를 **Specific QA ✓ / Abstract QA ✗** 로 분류 → 우리 실측(single/multi/relat 최상위, global 0.25)과 정확히 일치. HippoRAG 연산자 = `Link+PPR`(우리 심층설명 확인).
- **수치 수렴(Table 5)**: HotpotQA에서 **HippoRAG 50.3 ≈ VanillaRAG 50.8**(압도 없음) → 우리 외부검증(HippoRAG 클러스터 묻힘)과 동일. 논문 원문 *"graph-based RAG typically higher than vanilla, but if retrieved elements irrelevant, may degrade"* = 우리 "그래프 우위는 조건부".
- **결론 수렴**: 논문 *"no single method dominates across question types and cost → task-specific optimization"* = 우리 **§11 용도별 2 프로필**.
- **★ 함의**: **완전 독립 팀·다른 하니스(GPT-4o·BGE-M3·11벤치)가 우리와 같은 결론** → §10.6 "공식 SOTA 대조 없음" 갭 **부분 해소**, 우리 결론의 external validity↑("과적합 우연" 아님).
- **새 발견**: Table 5에서 **specific QA 최고가 대부분 RAPTOR**(트리 계층요약, 비그래프). 계층 요약이 핀포인트에도 강함 → **다음 후보 RAPTOR**(§9).

## 11. 사용 프로필 (범용 vs 커뮤니티)
> 결론은 "하나의 승자"가 아니라 **용도별 두 프로필**(§10.7~10.9). 상세·명령: [docs/RESULT.md](docs/RESULT.md).

| | 범용 | 커뮤니티(조직 특화) |
| --- | --- | --- |
| 기법 | `hybrid`(벡터+BM25) | `graphrag_e2b_hybrid`(그래프+벡터) / global 많으면 `graphrag_e2b_adaptive` |
| 언제 | 일반 QA·핀포인트·다중홉, 연결 약함 | 조직 문서가 실제 참조·연결 + global/종합 질의 |
| 그래프 추출 | 불필요·최고속 | 필요·느림 |
| config | `config/profile_general.yaml` | `config/profile_community.yaml` |

- **범용 근거**: 내부 judge 0.806(공동1위)·최고속, 외부 HotpotQA 동률.
- **커뮤니티 근거**: global서 그래프 역전(0.375>0.25), 외부 다중홉서 평면과 동급(격차 0.42→0.02).
- **주의(§10.9)**: 조직 문서가 자동생성 near-duplicate면 연결 이득↓ — **진짜 문서**일수록 커뮤니티 프로필 값어치↑.

## 부록 A. 논문 · 공식 구현 레퍼런스
> 2026-06 웹 검증 완료. 구현 시 각 논문·레포의 알고리즘을 1차 근거로 사용한다.

| 기법 | 논문 | arXiv | 공식 구현 |
| --- | --- | --- | --- |
| Standard RAG | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020) | [2005.11401](https://arxiv.org/abs/2005.11401) | — |
| GraphRAG | Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* | [2404.16130](https://arxiv.org/abs/2404.16130) | [microsoft/graphrag](https://github.com/microsoft/graphrag) |
| LightRAG | Guo et al., *LightRAG: Simple and Fast Retrieval-Augmented Generation* | [2410.05779](https://arxiv.org/abs/2410.05779) | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) |
| HippoRAG | Gutiérrez et al., *HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs* (NeurIPS 2024) | [2405.14831](https://arxiv.org/abs/2405.14831) | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| HippoRAG2 | Gutiérrez et al., *From RAG to Memory: Non-Parametric Continual Learning for LLMs* (ICML 2025) | [2502.14802](https://arxiv.org/abs/2502.14802) | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| LeanRAG | *LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval* (2025) | [2508.10391](https://arxiv.org/abs/2508.10391) | [RaZzzyz/LeanRAG](https://github.com/RaZzzyz/LeanRAG) |
| CausalRAG | Wang et al., *CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation* (ACL 2025 Findings) | [2503.19878](https://arxiv.org/abs/2503.19878) | [hippoley/CausalRAG](https://github.com/hippoley/CausalRAG) |
| HugRAG | *HugRAG: Hierarchical Causal Knowledge Graph Design for RAG* (2026) | [2602.05143](https://arxiv.org/abs/2602.05143) | 공개 레포 미확인 |

**관련 벤치마크 · 서베이**
- **HolisQA** — HugRAG 논문이 제안한 관계·전체이해 중심 QA 평가셋. 공개 형태·라이선스는 논문/레포에서 확인 후 채택 검토(§8).
- **In-depth Analysis of Graph-based RAG (VLDB 2025)** — [2503.04338](https://arxiv.org/abs/2503.04338) · [github/JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG). 12기법×11데이터셋 통합 벤치. 우리 결론과 수렴(§10.11) — 외부 타당성 근거.

# 작업 기록 (타임라인)

> 날짜별 진행 기록. 결론은 [RESULT.md](RESULT.md), 다음 할 일은 문서 하단 TODO.

---

## 작업 타임라인

> RAG 비교·검증 프로젝트(`ragbench`)의 진행 기록. 시각은 파일 최종수정 기준 근사치이며, 대부분 2026-06-29 저녁 ~ 06-30의 연속 세션이다.

### 2026-06-29 (저녁)
**설계 방향 확정 + 스캐폴딩 시작**
- CLAUDE.md를 **RAG 비교·검증 프로젝트**로 정의(처음엔 다른 프로젝트로 오해 → 신규 RAG → 문서 Q&A → 7개 기법 벤치마크로 확정).
- 비교 대상 7기법·임베딩 축·검증표 정리, **논문 부록 웹 검증**(arXiv·공식 레포).
- 공통 인터페이스 `core/interface.py` 작성. *(21:23)*

### 2026-06-30 (새벽 00~02시)
**하니스 + 베이스라인 + 코퍼스**
- 패키지/`pyproject` *(00:47)*, **Standard RAG** *(00:48)*, uv 환경.
- 공급자: Claude+OpenAI → **Google AI Studio(Gemini)** 전환, `GEMINI_API_KEY`, end-to-end 통과.
- **eval 하니스**(recall@k 등) + 결과 저장.
- 「주식회사 하울」 코퍼스 구축(동화 인물; 규정·행정·부서별·양식·통계·교육 + 채용/평가/급여/정보보안/출장/사업계획/인수인계).
- 코퍼스 생성기 `generate_large_corpus.py` *(01:56)*, **평가셋** `eval_sample.yaml` *(02:15)*.
- 검색 3종(standard/bm25/hybrid) 비교, 노이즈 토글, 유형 태그(single/multi/relational/global).

### 2026-06-30 (새벽 04시)
**코퍼스 품질 개선**
- "뼈대 문서" 지적 → **LLM으로 rich 본문 생성** `generate_rich_corpus.py` *(04:07)*, 153종 (≈$0.32).

### 2026-06-30 (오후 12~16시)
**GraphRAG + 로컬 임베딩 + 진단**
- GraphRAG 논문 정독·번역([GUIDE.md](GUIDE.md)), 구현 `methods/graphrag.py` *(12:21)* — MAX_TOKENS·nested async·임베딩 quota 해결.
- **로컬 임베딩**(multilingual-e5-small, **GPU**) 추가 → rate limit 해방.
- GraphRAG 공정 재대결(embed_kg_nodes=True + 로컬), 결과 `results/graphrag.json` *(15:47)*.
- ⭐ **그래프 진단**: 엔티티 노드에 **속성·설명·임베딩 없음** 발견 → "빈약한 그래프"가 저성능 원인.
- 그래프 품질 개선 설계안 [DESIGN.md](DESIGN.md) 작성.

### 2026-07-01 (그래프 품질 개선 — E2B + 레버 L1~L5 + Neo4j)
**빈약 그래프 → 풍부한 그래프 재구성 + 검색전략·전역강화 실험. 전 과정 로컬·무료(M2 gemma + RTX e5 + Neo4j).**
- **L1 추출기 교체**: `graphrag_e2b`(gemma4:e2b 산문추출 `이름|유형|설명` → 파싱). 210문서 3.5h → 엔티티 1,152·관계 1,376·**description 711개**(빈약 그래프는 0%).
- **L2 정규화**: 라벨/타입/이름 정규화(`normalize_type/rel/name`) 추출기 내장. 「」 병합·관계 표면형 통합. (기존그래프 수동병합은 스키마 파손 → 되돌림, 백업 `.pre_l2`)
- **L3 임베딩 점검**: 엔티티 전부 e5 임베딩 보유 확인 → **이미 정상**.
- **1차 4종 평가**(로컬 judge): global judge에서 **graphrag_e2b 1위(0.63)** — GraphRAG "global sensemaking 강점" 첫 실측. (핀포인트는 평면검색 우위)
- **재순위(E5Rerank)**: 동점 블롭 해소 → recall 0.13→0.34·relational judge 0.4→0.8 ✅ / **global judge 0.63→0.25 ❌**(top-4 컷이 breadth 죽임).
- **num_ctx 실측**: Ollama 기본 4096 절단 확정. 8GB에서 ~11~16k 가능하나 속도 대가. num_ctx=8192로도 **global 회복 실패**(0.125) → 병목은 컨텍스트 크기 아닌 "문서 4개만 투입".
- **L5 커뮤니티 요약**(`graphrag_e2b_l5`): Neo4j GDS Louvain 33군집 → gemma 요약 → 검색 주입. global **kw_recall 0.55→0.635 회복**(breadth 살아남) 하지만 **judge 0.25 그대로**(요약 노이즈).
- **Neo4j+Cypher+GDS 도입**: `docker-compose.yml`, 그래프 적재(`migrate_graph_to_neo4j.py`), 시각화·PageRank/Louvain 준비.
- **발견한 한계**: 추출기 `text[:2000]` 절단이 청크 2048과 불일치 → 긴 문서 손실(속도 위해 청크 키우며 놓침).

#### 미해결 결정 (다음 세션)
1. **검색전략**: global엔 breadth 유지 필요 → **적응형 top_n**(핀포인트=재순위 top4 / global=넓게) 확정 필요. (~12분 테스트)
2. **재구축**: truncation 수정(`text[:2000]`→넉넉히) + 청크 재설정(2048유지 vs 1024) + L5 → **한 번의 깨끗한 재구축** 후 전체 36문항 최종 비교.
3. CLAUDE.md §10.4에 1차 평가 기록됨. 재순위·num_ctx·L5·Neo4j는 추후 반영.

#### 인프라 상태 (재개용)
- Neo4j 실행 중(`docker compose up -d`, neo4j/ragbench123, :7474/:7687), 그래프 적재됨.
- 신규 코드: `methods/graphrag_e2b.py`(E2B추출·정규화·E5Rerank·L5), `config/ollama.yaml`(num_ctx 8192, chunk 2048), `scripts/{migrate_graph_to_neo4j,build_community_summaries}.py`.
- 결과: `results/local_eval/*_judge.json`, `storage/graphrag_e2b/community_summaries.json`(33요약).
- **비용 오늘 $0**(전부 로컬). Gemini 선불 잔액 유지.

### 2026-07-02 (v2 재구축 + 공정 최종비교)
**청크 1024 + truncation 수정으로 그래프 재구축 → baseline까지 동일 청크로 재평가(공정성 확보).**
- v2 그래프: 노드 1,152→2,310·커뮤니티요약 33→79(더 촘촘).
- **공정성 수정**: baseline이 예전(청크2048) 수치였음 → 3종 전부 청크1024 재인덱싱+재평가.

**최종 비교 (전부 청크1024·로컬 e5+gemma·36문항 judge)**

| 방식 | recall | kw_recall | judge | latency |
| --- | --- | --- | --- | --- |
| bm25 | 0.708 | 0.785 | 0.639 | 39s |
| **standard** | 0.863 | 0.840 | **0.750** | 38s |
| hybrid | 0.863 | 0.877 | 0.722 | 42s |
| graphrag_e2b | 0.321 | 0.593 | 0.500 | 75s |
| graphrag_e2b_l5 | 0.393 | 0.701 | 0.528 | 77s |

**유형별 judge_correct** (★ global):

| 방식 | single | multi | relational | global |
| --- | --- | --- | --- | --- |
| bm25 | 0.73 | 0.67 | 1.00 | 0.25 |
| standard | 1.00 | 0.83 | 0.80 | 0.25 |
| hybrid | 1.00 | 0.75 | 0.80 | 0.25 |
| graphrag_e2b | 0.55 | 0.42 | 1.00 | 0.25 |
| **graphrag_e2b_l5** | 0.36 | 0.67 | 0.80 | **0.38** |

- **결론**: 청크1024에서 **평면검색이 더 강해짐**(standard recall 0.69→0.86, judge 0.75) — 공정 재평가로 comparison이 실제로 바뀜. 그래프는 핀포인트 평가서 최하위. **단 graphrag_e2b_l5가 global 유일 승자(0.38 vs 전부 0.25)** — 공정 조건에서도 커뮤니티 요약의 전역 가치 생존.
- 트레이드오프 확정: **L5는 multi·global↑ / single↓** → **적응형 검색**(single=재순위 / global=L5·breadth)이 다음 스텝.
- 결과: `results/local_eval/{bm25,standard,hybrid,graphrag_e2b,graphrag_e2b_l5}_v2_judge.json`. 비용 $0.

### 2026-07-02 (심화 — 강judge 검증 → 버그 발견 → 검색층 하이브리드로 그래프 부활)
**"GraphRAG는 왜 지는가"를 5단계로 파고들어, 결국 "제대로 구현하면 이긴다"를 실측.**
1. **강judge 검증**: gemma judge가 그래프 점수를 부풀림 의심 → **Gemini judge로 재채점**. graphrag_e2b_l5 global 0.38→**0.00** 붕괴 → 약judge가 그래프를 관대하게 봤음이 드러남.
2. **강추출 검증**: "약한 gemma 추출 탓" 가설 → **Gemini로 전체 재추출**(desc 62%→**98%**, 타입 깨끗). 그런데도 그래프 judge 0.25~0.36으로 여전히 낮음 → "구조적 한계"로 **잠정 오결론**.
3. **버그 발견(사용자 반증)**: "버그 아니야?" → 진단하니 **재랭커 임베딩 버그** — 검색된 노드 내용이 `트리플 + 원문`인데 e5(512토큰)가 **앞의 트리플만 임베딩**해 정답 청크(근로기준법)를 낮게 매김. 정답이 풀에 있는데 top-k에서 탈락. → 재순위 시 트리플 제거하도록 수정(부분 회복).
4. **핵심 통찰(사용자)**: "버그가 전부 아니다 — 검색이 그래프 단일 기법이라 낮다"(의미/키워드 단일 대비 하이브리드가 이겼던 것과 동형).
5. **검색층 하이브리드**([methods/graphrag_e2b.py](src/ragbench/methods/graphrag_e2b.py) `GraphRAGE2BHybrid`): 그래프 검색 + **직접 청크 벡터검색**(standard 인덱스 재사용) **RRF 융합** → **graphrag_e2b 0.333 → hybrid 0.806**(2.4배), single 1.0·multi 0.92·relational 0.80로 **평면 수준 정밀도 회복** + global 0.38로 **연결성 강점 유지**.

**최종 비교 (Gemini 그래프·생성·judge, 버그수정 반영)**

| 방식 | judge | recall | single | multi | relat | global |
| --- | --- | --- | --- | --- | --- | --- |
| standard (평면) | 0.833 | 0.863 | 1.00 | 0.92 | 1.00 | 0.38 |
| graphrag_e2b (그래프만) | 0.333 | 0.250 | 0.36 | 0.50 | 0.20 | 0.12 |
| **graphrag_e2b_hybrid (그래프+직접벡터)** | **0.806** | 0.833 | 1.00 | 0.92 | 0.80 | 0.38 |

- **결론**: **GraphRAG 저성능 = 방법론의 구조적 한계가 아니라 ① 재랭커 임베딩 버그 + ② 단일 검색 기법.** 올바른 검색층(그래프+벡터 하이브리드 + 정상 재랭커)이면 평면 정밀도를 따라잡으며 전역·연결성을 더한다.
- 결과: `results/local_eval/*_gemini_{judge,fixed}.json`, `graphrag_e2b_hybrid_gemini.json`. 비용: Gemini 추출(thinking on)~2천원대 + 평가·재채점 소액.

### 2026-07-04 (gemma 무료 재현 + 방법론 위치 기록)
**버그수정된 코드로 gemma(로컬·무료) 재검증 → 하이브리드 효과가 모델 무관하게 재현됨.**

| 환경 | graph-only → hybrid (judge) | recall(hybrid) |
| --- | --- | --- |
| Gemini | 0.333 → 0.806 (2.4×) | 0.833 |
| **gemma** | 0.444 → 0.667 (1.5×) | **0.833** |

- **핵심**: 하이브리드 > graph-only가 **두 환경 모두 재현** → "검색층 하이브리드가 답"이 Gemini 아티팩트 아님.
- **검색 개선(recall 0.45→0.83)은 두 환경 거의 동일** — 검색은 e5(같은 임베딩)라 모델 무관. 차이나는 judge 최종값(gemma 0.667/Gemini 0.806)은 생성·판정 모델 강도 탓.
- 결과: `results/local_eval/{graphrag_e2b,graphrag_e2b_hybrid}_gemma_fixed.json`. 비용 $0.
- **방법론 위치**: 우리 검증이 "논문식인가" 정리 → CLAUDE.md §10.6(통제·애블레이션·강judge 교차검증·반증은 논문식 / 소규모 가상 코퍼스·유의성 부재는 논문 수준 아님 / 근본은 한국어·사내문서 도메인에 맞는 공개 벤치 부재).

### 2026-07-12 (연속 세션 — 공개벤치 외부검증 + HippoRAG2)
**시도·완료한 것**
- **HotpotQA 공개벤치**(n=100 다중홉) 도입 — 평면 3종 + 그래프 계열 외부검증. 중립 judge로 교정.
- **전 10기법 매트릭스**(회사 n=36) + run1/run2 **재현성**(상위4종 Δ=0). `hybrid=e2b_hybrid` 공동1위 0.806.
- **격차 원인분석**(§10.8): 연결성 아님(역설 — 회사가 더 촘촘한데 더 약함), 질문-정렬·답변스타일이 원인.
- **코퍼스 품질 실측**(§10.9): 회사 88%가 자동생성 near-duplicate(TTR 0.156). 내부 벤치 타당성 부분 흔들림.
- **용도별 2 프로필**(§11): 범용(hybrid) / 커뮤니티(graph+vector). config·docs/RESULT.md.
- **HippoRAG2 통합**(§10.10) — 공식 `hipporag` 어댑터([methods/hipporag.py]) + **e5 OpenAI호환 서버**([scripts/e5_openai_server.py])로 임베딩 통제(PPR만 변수). 내부 judge **0.806 공동1위** → "그래프 저성능=구현 성숙도" 확증.
- **HippoRAG2 외부검증**(§10.11): HotpotQA 0.640 — **25% recognition-memory JSON 실패**. 원인 판명: 트리플필터 프롬프트가 **Llama-3.3-70B DSPy튜닝**인데 Gemini로 돌림 + **e5 vs NV-Embed-v2 핸디캡**.
- **독립 논문 대조**: VLDB 2025([2503.04338]) 12기법 벤치와 결론 수렴 → 외부 타당성(§10.6 갭 부분해소).
- **논문 본문 정독**: v1(2405.14831 full — PPR·node specificity sᵢ=\|Pᵢ\|⁻¹·τ0.8·damping0.5), v2(2502.14802 §3 — Query-to-Triple, recognition memory=트리플 필터, passage 노드 contains edge, PPR 시드=구+passage).
- 아티팩트 2종: 검증 매트릭스 히트맵 · HippoRAG2 해설.

#### ★ 내일 할일 (TODO)
1. **§10.11 교정(먼저)** — "HotpotQA서 HippoRAG≈Vanilla"는 **VLDB의 v1(Llama-3-8B)** 기준임을 명시. **v2 논문 자체 수치는 HippoRAG2가 HotpotQA F1 75.5·recall@5 96.3 최고**(Llama-3.3-70B+NV-Embed). "압도 못 함"은 버전·모델 의존적 → 문장 분리.
2. **HippoRAG2 외부 깨끗 재실행** — 25% 버그 해소.
   - **07-13 새벽 시도 결과(실측)**: 어댑터에 내부LLM override 추가(config `hipporag_llm_{base_url,model,key_env}`) → **NIM Llama-3.3-70B 배선 성공**. **NV-Embed-v2는 무료티어 없음**(v1·nv-embedqa-e5-v5만). ★ **HippoRAG는 OpenIE·fact를 `{LLM명}` 네임스페이스로 저장 → LLM 교체 시 전체 재인덱싱 필수**(인덱스 재사용 불가, 실측 확인). ★ **NIM 무료 40 RPM이 bulk OpenIE에 치명적** — 30문서 재인덱싱도 RateLimitError 연발, 문서당 24~62초. 991문서는 비현실적.
   - **✅ 07-13 완료(반증) — 0.640은 진짜 수치.** ① JSON-repair 추가 → 파싱에러 25→0, judge 0.640 **불변** ② nv-embed-v1(4096d) 재인덱싱 → 0.650(노이즈). **두 가설 다 반증** → "25% 저하 하한"은 틀렸고 0.640~0.650이 HippoRAG2 진짜 성능. §10.11 교정 완료. **이 TODO 종결.**
3. **RAPTOR 통합 (다음 기법 1순위) — 07-13 출처검증 완료, 착수 준비됨.**
   - **정의**: 재귀 요약 트리(비그래프). 청크→임베딩→클러스터링(GMM soft + UMAP)→클러스터를 LLM 요약=부모노드→루트까지 재귀. 검색: tree traversal / collapsed tree. arXiv 2401.18059, [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor).
   - **우리에 유리**: `BaseEmbeddingModel`/`BaseSummarizationModel`/`BaseQAModel` 상속으로 **e5·Gemini 그대로 주입**(통제 비교 유지, 기본은 OpenAI). API: `RetrievalAugmentation.add_documents(text)`/`.answer_question(q)`/`.save()`.
   - **착수 순서**: ① 설치(clone+requirements, pip패키지 여부 확인) ② `methods/raptor.py` 어댑터(커스텀 e5/Gemini 3모델 → RetrievalAugmentation; index=add_documents+save, query=answer_question+컨텍스트매핑) ③ registry+config ④ 평가(회사 핀포인트 + HotpotQA — VLDB의 "specific QA 최강" 재현되나).
   - **가설**: 계층 요약이 핀포인트에도 강한지(§10.11 새 발견). HippoRAG(PPR=옆으로 연상)와 다른 축(위로 추상화) 비교.
4. (선택) LightRAG 추가 / 논문 "new variants(연산자 재조합) ↔ 우리 e2b_hybrid" 대조.
5. **push** — 밀린 커밋 원격 반영 확인(자격증명은 로컬).

### 핵심 결과 (누적)
- **하이브리드 검색이 대체로 최고**, 다중 홉·노이즈가 공통 난점.
- **GraphRAG는 현재 미결론** — 그래프가 빈약(노드 속성/임베딩/커뮤니티 요약 없음)해 공정 판정 불가(CLAUDE.md §10).
- 비용 누적 ~$5~7 (대부분 생성 LLM, 임베딩은 로컬로 $0).

### 산출물 위치
- 설계·규칙·결과 요약: `CLAUDE.md`
- 논문 정리: `docs/GUIDE.md`
- 그래프 개선 설계: `docs/DESIGN.md`
- 결과 데이터: `results/*.json` (+ `results/archive_24doc/`)
- 코퍼스: `data/company/` (+ `data/noise/`)
- 생성기: `scripts/generate_large_corpus.py`, `scripts/generate_rich_corpus.py`

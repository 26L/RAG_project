# 결과 — RAG 비교·검증 결론

> RAG 기법 비교의 **결론**. 과정은 [LOG.md](LOG.md), 검증 설계는 [VALIDATION.md](VALIDATION.md), 상세 근거는 [../CLAUDE.md](../CLAUDE.md) §10.

---

## 핵심 압축 (한 장 결론)

> 여러 RAG 기법을 **같은 코퍼스·같은 임베딩·같은 생성·같은 judge**로 두고 **검색 메커니즘만 변수**로 비교한 결과의 압축. 과정·타임라인은 [LOG.md](LOG.md), 상세 근거는 [../CLAUDE.md](../CLAUDE.md) §10.

### 1. 무엇을 비교했나
- **평면 검색**: standard(벡터) · bm25(키워드) · hybrid(벡터+BM25 RRF)
- **그래프 검색**: GraphRAG 계열(자체 e2b·평면트리플·스키마·동적) · **HippoRAG2**(공식, OpenIE+PPR)
- 다음: RAPTOR(트리 계층요약) *(착수점 확정)*
- 전부 **공통 인터페이스**(index/query) 뒤 플러그형 백엔드. 평가: LLM-judge(중립) 중심.

### 2. RAG 핵심 결론 (압축)
1. **단일 승자 없다 — 용도별.** 핀포인트/다중홉엔 하이브리드·성숙한 그래프, 종합(global)엔 그래프 계열. → **2 프로필**([RESULT.md](RESULT.md)).
2. **하이브리드가 전천후 최선**(내부 judge 0.806 공동1위·최고속). 단 **"하이브리드>단일검색"은 도메인 특이적** — 한국어 사내문서에선 이겼지만 HotpotQA(위키)에선 의미검색과 동률(McNemar 무유의차).
3. **그래프 저성능 = 방법론 아니라 구현 성숙도.** 자체 그래프(e2b 0.39) vs 공식 **HippoRAG2 PPR 0.806**(하이브리드와 동률) — 같은 임베딩(e5)에서. OpenIE+PPR+passage노드+node specificity가 격차의 정체.
4. **그래프 강점 = 다중홉(PPR 연상) + 전역 sensemaking(커뮤니티 요약).** 약점 = 핀포인트(무랭킹 대량 반환이 노이즈).
5. **연결성 ↑ ≠ 성능 ↑ (반직관).** 과연결 그래프가 핀포인트엔 오히려 독. 시노님 20배 늘려도 불변. → **질문이 실제로 연결을 요구할 때만** 그래프가 값어치.

### 3. RAG 벤치마킹 방법론 (재사용 규칙)
- **통제**: 임베딩·생성·judge 고정, **기법만 변수**. (임베딩 다르면 결과 해석 불능)
- **judge > recall@k**: 그래프는 무랭킹 반환이라 recall@k가 구조적 불리 → LLM-judge로 채점.
- **공개벤치 교차검증**(HotpotQA): 자체 코퍼스 과적합·과잉일반화 방지. 독립 VLDB 벤치와 결론 수렴 확인.
- **코퍼스 품질 실측**: 자체 코퍼스 88%가 자동생성 near-duplicate(TTR 0.156)였음 → 수치에 아티팩트 혼입. 실무 신뢰는 외부·손작성 위주.
- **자기 반증**: "그래프=구조적 한계", "HippoRAG 0.640=버그로 저하" 두 결론을 실측으로 **뒤집음**. 그럴듯한 핑계를 걷어내는 게 검증의 값어치.

### 4. 실무 선택 (2 프로필)
| | 범용 | 커뮤니티(조직 특화) |
| --- | --- | --- |
| 기법 | `hybrid` | `graphrag_e2b_hybrid` / global 많으면 `adaptive` |
| 언제 | 일반·핀포인트·다중홉, 연결 약함 | 실제 참조·연결된 조직문서 + 종합 질의 |
| 그래프 추출 | 불필요·최고속 | 필요·느림 |

### 5. 한 줄
**RAG는 "검색 방식 × 질문 유형 × 문서 성격"의 매칭 문제다.** 하이브리드가 안전한 기본값, 그래프는 질문이 진짜 연결·종합을 요구할 때. 그래프의 성패는 연결성의 양이 아니라 **구현 성숙도 + 질문-그래프 정렬**에 달렸다.

---

## 사용 프로필 — 범용 vs 커뮤니티

> 벤치마크 결론은 **"하나의 승자"가 아니라 용도별 두 프로필**이다(§10.7~10.9).
> 질문이 연결(다중홉·종합)을 실제로 요구하는지, 문서가 진짜 연결돼 있는지로 갈린다.

### 한눈에

| | **범용 프로필** | **커뮤니티(조직 특화) 프로필** |
| --- | --- | --- |
| 추천 기법 | `hybrid` (벡터+BM25 RRF) | `graphrag_e2b_hybrid` (그래프+벡터) / global 많으면 `graphrag_e2b_adaptive` |
| 언제 | 일반 QA, 핀포인트·다중홉, 문서 연결 약함 | 특정 조직/커뮤니티의 **연결된** 문서, 종합·global·연결성 질의 |
| 그래프 추출 | 불필요 | 필요(비용·시간↑) |
| 속도 | 최고속(~1.3s) | 느림(~3s, 추출 별도) |
| config | `config/profile_general.yaml` | `config/profile_community.yaml` |

### 프로필 A — 범용

**언제**: 문서 간 연결이 약하거나, 질문이 대부분 특정 사실/2홉이면. 대부분의 실무가 여기 해당.

**근거**
- 내부 매트릭스 judge **0.806(공동 1위)**, single/multi/relational 최고, **최고속**.
- 외부 HotpotQA(다중홉 n=100)에서도 평면과 **통계적 동률**(McNemar 무유의차).
- 그래프 추출 비용 0, 도메인 로버스트.

**실행**
```bash
ragbench index --method hybrid --config config/profile_general.yaml --data <corpus>
ragbench eval  --method hybrid --config config/profile_general.yaml --eval-set <set> --judge
```

### 프로필 B — 커뮤니티(조직 특화)

**언제**: 한 조직/커뮤니티의 문서가 서로 **실제로 참조·연결**되고, "전사적으로/각 부서가/어떻게 이어지나" 같은 **종합·global·연결성** 질문이 중요할 때.

**근거**
- **global 유형에서 그래프 계열이 역전**(dynamic/e2b/adaptive 0.375 > 평면 0.25) — sensemaking 강점.
- 외부 다중홉에서 그래프만 0.61~0.67로 **평면과 동급**(그래프-평면 격차 0.42→0.02 붕괴) → 연결이 실제 필요한 질의에 강함.
- `graphrag_e2b_hybrid`는 그래프+직접벡터라 **핀포인트도 유지**(내부 0.806)하며 연결성을 더함.

**주의(§10.9)**: 조직 문서가 **자동생성 near-duplicate**면 연결이 밀도만 높고 의미 없어 이득이 준다. **사람이 쓴 진짜 문서**일수록 이 프로필의 값어치가 산다.

**실행** (그래프 3단계: 추출 → 요약 → 평가)
```bash
ragbench index --method graphrag_e2b --config config/profile_community.yaml --data <corpus>
python scripts/build_community_summaries.py config/profile_community.yaml storage/graphrag_e2b
ragbench eval --method graphrag_e2b_hybrid --config config/profile_community.yaml --eval-set <set> --judge
```

### 선택 기준 (결정 표)

| 상황 | 프로필 |
| --- | --- |
| 질문이 특정 사실/2홉, 문서 연결 약함 | 범용(`hybrid`) |
| 조직 문서가 서로 참조·연결 + global/종합 질의 | 커뮤니티(`graphrag_e2b_hybrid`) |
| global·"전사/각 부서/어떻게 이어지나" 가 다수 | 커뮤니티(`graphrag_e2b_adaptive`) |
| 속도·비용 최우선, 그래프 부담 | 범용(`hybrid`) |

> 근거 수치: [../CLAUDE.md](../CLAUDE.md) §10.4~10.9. 실행 상세: [GUIDE.md](GUIDE.md).

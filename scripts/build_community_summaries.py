"""L5 — Neo4j 그래프에서 Louvain 커뮤니티를 탐지하고 각 군집을 gemma로 요약.

GraphRAG의 전역(sensemaking) 엔진. 원문 대신 "군집 요약"(압축)을 검색에 쓰면
온디바이스 컨텍스트 제약(num_ctx) 안에서 넓은 breadth를 확보한다.

사전: docker compose up -d (Neo4j+GDS), 그래프 적재(scripts/migrate_graph_to_neo4j.py).
출력: storage/<method>/community_summaries.json  [{id, size, members, summary}]

사용:
    OLLAMA_BASE_URL=http://<M2>:11434 \
    .venv/bin/python scripts/build_community_summaries.py [config/ollama.yaml] [storage/graphrag_e2b]
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

from ragbench.core.config import Config
from ragbench.llms.factory import build_llm

load_dotenv("/home/rokey/your-project/.env")  # GEMINI_API_KEY 등

CFG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config/ollama.yaml"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "storage/graphrag_e2b"
MIN_SIZE = 3  # 이보다 작은 군집은 요약 생략(고립 노드 노이즈)

URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PWD = os.environ.get("NEO4J_PASSWORD", "ragbench123")

PROMPT = (
    "다음은 한 회사 문서 지식그래프의 한 군집(연결된 엔티티들)이다. "
    "이 군집이 '무엇에 관한 것인지'를 3~4문장으로 요약하라. "
    "핵심 주체·규정·프로젝트와 그들의 관계를 포함하되, 사실만 간결히 쓴다.\n\n"
    "[엔티티]\n{entities}\n\n[관계]\n{relations}\n\n요약:"
)


def main() -> None:
    cfg = Config.load(CFG_PATH)
    llm = build_llm(cfg)
    drv = GraphDatabase.driver(URL, auth=(USER, PWD))

    def run(cypher, **kw):
        with drv.session() as s:
            return list(s.run(cypher, **kw))

    # 1) Louvain → 노드에 communityId 기록
    run("CALL gds.graph.drop('g', false) YIELD graphName")
    run(
        "CALL gds.graph.project.cypher('g',"
        " 'MATCH (n) WHERE n.name IS NOT NULL RETURN id(n) AS id',"
        " 'MATCH (a)-[r]->(b) WHERE a.name IS NOT NULL AND b.name IS NOT NULL"
        "  RETURN id(a) AS source, id(b) AS target') YIELD nodeCount"
    )
    run("CALL gds.louvain.write('g', {writeProperty: 'community'}) YIELD communityCount")

    # 2) 커뮤니티별 멤버 크기
    comms = run(
        "MATCH (n) WHERE n.community IS NOT NULL "
        "RETURN n.community AS c, count(*) AS sz ORDER BY sz DESC"
    )
    targets = [r["c"] for r in comms if r["sz"] >= MIN_SIZE]
    print(f"커뮤니티 {len(comms)}개 중 크기{MIN_SIZE}+ = {len(targets)}개 요약 시작", flush=True)

    out = []
    for i, cid in enumerate(targets, 1):
        members = run(
            "MATCH (n) WHERE n.community=$c "
            "RETURN n.name AS name, labels(n) AS labels, n.description AS desc LIMIT 40",
            c=cid,
        )
        rels = run(
            "MATCH (a)-[r]->(b) WHERE a.community=$c AND b.community=$c "
            "RETURN a.name AS s, type(r) AS rel, b.name AS t LIMIT 60",
            c=cid,
        )
        ent_txt = "\n".join(
            f"- {m['name']} ({m['labels'][0] if m['labels'] else '?'}): {m['desc'] or ''}"
            for m in members
        )
        rel_txt = "\n".join(f"- {r['s']} -[{r['rel']}]-> {r['t']}" for r in rels)
        summary = str(llm.complete(PROMPT.format(entities=ent_txt, relations=rel_txt))).strip()
        out.append(
            {
                "id": cid,
                "size": len(members),
                "members": [m["name"] for m in members],
                "summary": summary,
            }
        )
        print(f"  [{i}/{len(targets)}] c={cid} size={len(members)} → {summary[:50]}...", flush=True)

    path = os.path.join(OUT_DIR, "community_summaries.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"\n저장: {path} ({len(out)}개 요약)")
    drv.close()


if __name__ == "__main__":
    main()

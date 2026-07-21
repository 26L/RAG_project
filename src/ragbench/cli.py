"""ragbench CLI — 인덱싱/질의 진입점.

  ragbench index --method standard --data data
  ragbench query --method standard "질문 내용"
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .core.config import Config
from .embeddings.factory import build_embed_model
from .ingest.loader import load_documents
from .llms.factory import build_llm
from .registry import build_backend


def _make_backend(args):
    """CLI 인자로부터 설정·LLM·임베딩을 만들고 요청한 기법의 백엔드를 조립한다.

    입력: args — `--config`(설정 YAML 경로), `--method`(기법 이름)
    출력: (Config, RagBackend) 튜플
    """
    cfg = Config.load(args.config)
    llm = build_llm(cfg)
    embed = build_embed_model(cfg)
    return cfg, build_backend(args.method, cfg, llm, embed)


def cmd_index(args):
    """`index` 서브커맨드 — 문서 디렉토리를 로드해 해당 기법으로 인덱싱·영속화한다.

    입력: args — `--data`(문서 디렉토리, 없으면 cfg.data_dir) + 공통 인자
    출력: 없음(`storage_dir/<method>` 에 인덱스 저장, 진행 상황 출력)
    """
    cfg, backend = _make_backend(args)
    data_dir = args.data or cfg.data_dir
    docs = load_documents(data_dir)
    print(f"'{data_dir}' 에서 문서 {len(docs)}개 로드. '{args.method}' 기법으로 인덱싱 중...")
    backend.index(docs)
    print(f"인덱스 저장 완료 → {cfg.storage_dir}/{args.method}")


def cmd_query(args):
    """`query` 서브커맨드 — 질문 하나를 던져 답변과 출처를 출력한다.

    입력: args — `question`(질문 텍스트) + 공통 인자
    출력: 없음(답변 본문과 출처 목록을 stdout 에 출력)
    """
    _, backend = _make_backend(args)
    result = backend.query(args.question)
    print("\n=== 답변 ===")
    print(result.answer)
    print("\n=== 출처 ===")
    for i, ctx in enumerate(result.contexts, 1):
        print(f"[{i}] {ctx.source}  (score={ctx.score})")


def _fmt_agg(agg: dict) -> str:
    """집계 지표 딕셔너리를 한 줄 문자열로 포맷한다(값 없는 키는 생략).

    입력: agg — n·recall·precision·mrr·keyword_recall·judge_correct·latency_s 집계
    출력: "recall=0.8  mrr=0.7" 형태의 문자열
    """
    keys = ("n", "recall", "precision", "mrr", "keyword_recall", "judge_correct", "latency_s")
    return "  ".join(f"{k}={agg.get(k)}" for k in keys if agg.get(k) is not None)


def cmd_eval(args):
    """`eval` 서브커맨드 — 평가셋을 돌려 전체·유형별 지표를 집계하고 결과를 저장한다.

    입력: args — `--eval-set`(평가셋 YAML), `--judge`(LLM-as-judge 채점 여부) + 공통 인자
    출력: 없음(집계표를 출력하고 `results/<method>.json` 에 설정+리포트 저장)
    """
    import json
    import os

    from .eval.dataset import load_eval_set
    from .eval.harness import run_eval

    cfg, backend = _make_backend(args)
    judge_llm = build_llm(cfg) if args.judge else None
    items = load_eval_set(args.eval_set)
    print(
        f"평가셋 {len(items)}문항 · 기법 '{args.method}' · top_k={cfg.top_k}"
        f"{' · LLM-judge ON' if args.judge else ''} 실행 중..."
    )
    report = run_eval(backend, items, cfg.top_k, judge_llm=judge_llm)

    print("\n=== 전체 집계 ===")
    print(" ", _fmt_agg(report["aggregate"]))
    print("\n=== 유형별 ===")
    for qtype, agg in report["by_type"].items():
        print(f"  [{qtype:10s}] {_fmt_agg(agg)}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{args.method}.json")
    payload = {
        "method": args.method,
        "config": {"llm": cfg.llm.model, "embed": cfg.embed.model, "top_k": cfg.top_k},
        "report": report,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 → {out_path}")


def cmd_compare(args):
    """`compare` 서브커맨드 — `results/*.json` 을 모아 기법 비교표를 출력한다.

    입력: args — 사용하지 않음(결과 파일 전체를 파일명 라벨로 비교)
    출력: 없음(비교표를 stdout 에 출력. 결과 파일이 없으면 안내만 출력)
    """
    import glob
    import json

    paths = sorted(glob.glob("results/*.json"))
    if not paths:
        print("results/ 에 결과 파일이 없습니다. 먼저 'ragbench eval' 를 실행하세요.")
        return

    import os

    rows = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        label = os.path.splitext(os.path.basename(p))[0]  # 파일명 기준 라벨
        rows.append((label, d["report"]["aggregate"]))

    cols = ("recall", "precision", "mrr", "keyword_recall", "judge_correct", "latency_s")
    header = f"{'method':12s} " + " ".join(f"{c:>14s}" for c in cols)
    print(header)
    print("-" * len(header))
    for method, agg in rows:
        cells = " ".join(f"{str(agg.get(c)):>14s}" for c in cols)
        print(f"{method:12s} {cells}")


def main():
    """CLI 진입점 — .env 로드 후 서브커맨드(index/query/eval/compare)를 파싱해 실행한다.

    출력: 없음(선택된 서브커맨드 함수를 호출)
    """
    load_dotenv()
    parser = argparse.ArgumentParser(prog="ragbench")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/default.yaml")
    common.add_argument("--method", default="standard")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", parents=[common], help="문서 인덱싱")
    p_index.add_argument("--data", help="문서 디렉토리 (기본: config 의 data_dir)")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", parents=[common], help="질의")
    p_query.add_argument("question", help="질문 텍스트")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("eval", parents=[common], help="평가셋으로 지표 집계")
    p_eval.add_argument("--eval-set", default="config/eval_sample.yaml", help="평가셋 YAML")
    p_eval.add_argument("--judge", action="store_true", help="LLM-as-judge 정답 채점(비용 발생)")
    p_eval.set_defaults(func=cmd_eval)

    p_compare = sub.add_parser("compare", help="results/*.json 기법 비교표")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

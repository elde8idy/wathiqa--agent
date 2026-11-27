"""Command line interface.

    python -m wathqa run --file data/sample_docs/invoice_clean.txt
    python -m wathqa run --file data/sample_docs/invoice_vat_mismatch.txt --yes
    python -m wathqa eval
    python -m wathqa serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .evaluate import run_eval
from .graph import build_graph, resume_review, start_review


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    graph = build_graph(settings)
    result = start_review(
        document_text=args.text, document_path=args.file, settings=settings, graph=graph
    )
    thread_id = result["thread_id"]

    if result["status"] == "needs_review":
        state = result["state"]
        print("== Human review required ==")
        print("Flags:", ", ".join(state.get("flags", [])) or "-")
        for reason in state.get("decision_reasons", []):
            print(" -", reason)
        if state.get("investigation"):
            print("Investigation:")
            for entry in state["investigation"]:
                print(" -", entry)
        if args.yes:
            feedback = "approve"
            print("Auto-answering with 'approve' (--yes).")
        else:
            feedback = input("Approve or reject? [approve/reject]: ")
        result = resume_review(thread_id, feedback, settings=settings, graph=graph)

    state = result["state"]
    print()
    print(state.get("report", "(no report)"))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    report = run_eval()
    print(json.dumps(report["metrics"], indent=2))
    print("Full report written to eval_report.json")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("wathqa.api:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wathqa")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="review one document")
    run_p.add_argument("--file", help="path to a .txt/.md/.pdf document")
    run_p.add_argument("--text", help="document text passed directly")
    run_p.add_argument("--yes", action="store_true", help="auto-approve the human step")
    run_p.set_defaults(func=cmd_run)

    eval_p = sub.add_parser("eval", help="run the labelled evaluation set")
    eval_p.set_defaults(func=cmd_eval)

    serve_p = sub.add_parser("serve", help="start the FastAPI server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

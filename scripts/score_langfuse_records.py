"""Publish reproducible resource metrics and enrich existing judge scores."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.langfuse_mcp import LangfuseMCP
from src.tracing import get_client


def publish_resource_metrics(client, record):
    cost = record["cost"]
    metrics = {
        "synthesis_tokens": cost["total_tokens"],
        "total_agent_tokens": cost["total_tokens"] + cost["aux_prompt_tokens"] + cost["aux_completion_tokens"],
        "turn_wall_time_ms": record["duration_ms"],
        "refused": int(record["output"]["refused"]),
        "execution_error": int(record["error"] is not None),
    }
    for metric, value in metrics.items():
        sid = hashlib.sha256(f"{record['trace_id']}:{metric}".encode()).hexdigest()[:32]
        client.create_score(score_id=sid, trace_id=record["trace_id"], name=metric, value=float(value), environment="sdk-experiment", comment="Computed from the saved RunRecord; total_agent_tokens excludes judge calls.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_id")
    parser.add_argument("--arm", choices=["naive", "engineered", "both"], default="both")
    args = parser.parse_args()
    mcp = LangfuseMCP()
    client = get_client()
    for arm in (("naive", "engineered") if args.arm == "both" else (args.arm,)):
        path = Path("runs") / f"{args.base_id}-{arm}"
        records = [json.loads(line) for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        traces = [r["trace_id"] for r in records]
        existing = {}
        cursor = None
        while True:
            params = {"limit": 100, "traceId": traces}
            if cursor:
                params["cursor"] = cursor
            response = mcp.call("listScores", params)
            for score in response["data"]:
                existing[(score["subject"]["id"], score["name"])] = score["id"]
            cursor = response.get("meta", {}).get("cursor")
            if not cursor:
                break
        by_id = {r["record_id"]: r for r in records}
        judges = [json.loads(line) for line in (path / "judgements.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for judge in judges:
            if judge["score"] is None:
                continue
            trace = by_id[judge["record_id"]]["trace_id"]
            score_id = existing[(trace, judge["metric"])]
            client.create_score(score_id=score_id, trace_id=trace, name=judge["metric"], value=float(judge["score"]), comment=judge.get("reason", ""), environment="sdk-experiment")
        for record in records:
            publish_resource_metrics(client, record)
        client.flush()
        print(arm, "judge reasons restored and resource metrics published for", len(records), "turns", flush=True)


if __name__ == "__main__":
    main()

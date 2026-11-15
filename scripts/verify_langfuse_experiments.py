"""Audit local records against experiment items and live observations through MCP."""
import argparse
import json
from pathlib import Path
from scripts.langfuse_mcp import LangfuseMCP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_id")
    parser.add_argument("--require-scores", action="store_true")
    args = parser.parse_args()
    mcp = LangfuseMCP()
    experiments = mcp.call("listExperiments", {"fromStartTime": "2026-11-01T00:00:00Z", "limit": 100, "fields": ["core", "scores"]})
    Path(".cache/langfuse-experiments-verification.json").write_text(json.dumps(experiments, indent=2))
    expected_names = {f"{args.base_id}-{arm}" for arm in ("naive", "engineered")}
    selected = [e for e in experiments["data"] if e["name"] in expected_names]
    print("EXPERIMENTS", [(e["name"], e["itemCount"]) for e in selected], flush=True)
    assert len(selected) == 2, "Requested experiment pair is missing"
    for arm in ("naive", "engineered"):
        name = f"{args.base_id}-{arm}"
        path = Path("runs") / name
        if not (path / "manifest.json").exists():
            print(name, "still running", flush=True)
            continue
        records = [json.loads(line) for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(records) == 52 and all(r["trace_id"] and not r["error"] for r in records)
        items = mcp.call("listExperimentItems", {"fromStartTime": "2026-11-12T00:00:00Z", "fields": ["core", "dataset", "scores"], "limit": 100, "scoreLimit": 50, "experimentName": [name]})
        Path(f".cache/langfuse-{arm}-items.json").write_text(json.dumps(items, indent=2))
        assert len(items["data"]) == 52
        assert {i["traceId"] for i in items["data"]} == {r["trace_id"] for r in records}
        assert {i["experimentItemId"] for i in items["data"]} == {r["record_id"] for r in records}
        if args.require_scores:
            required = {"answer_correctness", "groundedness", "citation_support", "total_agent_tokens", "synthesis_tokens", "turn_wall_time_ms", "refused", "execution_error"}
            for item in items["data"]:
                assert required <= {s["name"] for s in item["scores"]}, item["experimentItemId"]
        print(name, "52 remote items match 52 local live trace IDs; scores checked:", args.require_scores, flush=True)


if __name__ == "__main__":
    main()

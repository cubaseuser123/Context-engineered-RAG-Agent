"""Check a revised judge against the assisted development review; preserve old scores."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ["DISABLE_LLM_CACHE"] = "1"

from evals.dataset import load_dataset
from evals.judges import judge_record, JUDGE_VERSION
from src.config import SCENARIOS_PATH
from src.models.run_record import RunRecord
from src.tracing import flush


def main():
    review = json.loads(Path("evals/datasets/context_v2_human_review.json").read_text(encoding="utf-8"))
    records = {}
    for run in ("smoke-v2-engineered-b5000-20261112_190504-engineered", "smoke-v2-engineered-b5000-20261112_191131-engineered"):
        for line in (Path("runs") / run / "records.jsonl").read_text(encoding="utf-8").splitlines():
            r = RunRecord.model_validate_json(line)
            records[r.trace_id] = r
    ds = load_dataset(SCENARIOS_PATH)
    results = []
    for item in review["items"]:
        if item["split"] != "development":
            continue
        r = records[item["trace"]]
        scenario = ds.by_id(r.scenario_id)
        j = judge_record(r, scenario, expected=scenario.expected_for(r.turn_index), metric="answer_correctness")
        expected = int(item["human"]["human_correctness"] == "Correct")
        results.append({"record_id": r.record_id, "human": expected, "judge": j.score, "reason": j.reason, "error": j.error})
        print(r.record_id, "human", expected, "judge", j.score, flush=True)
    result = {"judge_version": JUDGE_VERSION, "review_type": "Assisted development review; not independent calibration", "agree": sum(x["human"] == x["judge"] for x in results), "total": len(results), "items": results}
    Path("reports/context_v3_review_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    flush()
    print("Agreement:", result["agree"], "/", result["total"], flush=True)


if __name__ == "__main__":
    main()

"""Execute each turn inside its dataset experiment, preserving conversation order."""
from langfuse import propagate_attributes


def make_invoker(client, dataset_name, run_id, arm):
    dataset = client.get_dataset(dataset_name)
    items = {item.id: item for item in dataset.items}

    def invoke(app, state, scenario_id, turn_index):
        record_id = f"{scenario_id}:{turn_index:02d}"
        item = items[record_id]
        captured = {}

        def task(*, item, **kwargs):
            with propagate_attributes(
                session_id=f"{run_id}/{scenario_id}",
                trace_name=f"{arm} / {scenario_id} / turn {turn_index}",
                tags=[arm, "live-evaluation"],
            ):
                captured["state"] = app.invoke(state)
            return {"response": captured["state"].get("response", "")}

        result = client.run_experiment(
            name=run_id, run_name=run_id, data=[item], task=task,
            max_concurrency=1,
            metadata={"run_id": run_id, "arm": arm, "scenario_id": scenario_id,
                      "turn_index": turn_index, "record_id": record_id, "execution": "live"},
            description="Live API evaluation; sequential conversation turns; cache disabled.",
        )
        if "state" not in captured or len(result.item_results) != 1:
            raise RuntimeError(f"Experiment turn failed: {record_id}")
        trace_id = result.item_results[0].trace_id
        if not trace_id:
            raise RuntimeError(f"Missing experiment trace: {record_id}")
        return captured["state"], trace_id

    return invoke

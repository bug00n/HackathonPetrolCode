"""Recompute jury evidence from the pinned local dataset, without retraining."""

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.main import (  # noqa: E402
    history_interval_command,
    replay_v2_shadow_command,
    run_model_demo,
)
from source.ui_data import discover_ui_context, ui_hybrid_snapshot, ui_stage_snapshot  # noqa: E402

output = ROOT / "release/evidence"
output.mkdir(parents=True, exist_ok=True)
context = discover_ui_context(ROOT)
dataset = context.latest_dataset
model = context.forecast_artifacts[0].path
at = datetime.fromisoformat("2025-06-01T12:00:00+03:00")
report = {
    "release_id": context.release_id,
    "dataset": Path(dataset).name,
    "as_of": at.isoformat(),
    "source": "organizer data and explicit model scenarios",
}
report["demo"] = {}
for name in (
    "blend_normal",
    "blend_risk",
    "blend_t95_risk",
    "blend_cetane_risk",
    "blend_missing",
    "blend_tradeoff",
):
    r = run_model_demo(name, run_dir=output / "demo-runs")
    report["demo"][name] = r.model_dump(mode="json")
started = time.perf_counter()
report["interval"] = history_interval_command(
    dataset,
    model,
    at,
    datetime.fromisoformat("2025-06-01T18:00:00+03:00"),
    run_dir=output / "history-runs",
)
report["interval_seconds"] = time.perf_counter() - started
report["hybrid"] = asdict(ui_hybrid_snapshot(dataset, model, at))
report["v2"] = replay_v2_shadow_command(dataset, context.v2_artifacts[0].path, at)
report["stages"] = {
    page: asdict(ui_stage_snapshot(page, dataset, at)) for page in ("avt", "hydrotreating")
}
assert report["interval"]["points"] == 7
assert report["hybrid"]["status"] == "ready"
assert all(value["status"] == "ready" for value in report["stages"].values())
(output / "jury-evidence.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "points": 7,
            "interval_seconds": report["interval_seconds"],
            "hybrid": report["hybrid"]["constraint_status"],
        }
    )
)

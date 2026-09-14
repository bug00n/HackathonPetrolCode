# Stage 7: trusted history artifact serving

> Current status: Stage 7 connects a trusted local forecast artifact to the
> history backend path through CLI. It does not add an action model, train a new
> model, or enable industrial setpoint recommendations.

## Purpose

Stage 7 closes the gap between the prepared historical dataset and the model
artifact contract. A user can now run one history forecast cycle from a checked
prepared dataset and an explicitly trusted local model directory, then inspect
the same journal files used by the model-demo flow.

## What Is Implemented

`python -m source.main run-history`:

- loads a prepared dataset from `data/processed/<dataset_id>/` or another
  supplied directory;
- loads a local model artifact only when `--trusted-model` is present;
- verifies model metadata before `joblib` loading, including horizon,
  tag-dictionary hash, target signal and target unit;
- builds leakage-safe serving features in the exact order stored in artifact
  metadata;
- runs the existing `history` scenario through `run_cycle`;
- writes the normal journal files under `runs/` or `--run-dir`;
- prints the strict `Recommendation` JSON.

Example:

```bash
python -m source.main run-history \
  --dataset data/processed/<dataset_id> \
  --model artifacts/models/<model_id> \
  --trusted-model \
  --as-of 2026-01-15T09:00:00Z
```

The command is intentionally explicit about trust because local joblib artifacts
can execute code when loaded. A missing `--trusted-model` flag returns a CLI
error instead of silently loading the artifact.

## Limits

- Stage 7 serves a forecast artifact; it does not create or retrain one.
- A model with `supports_forecast=true` still does not imply
  `supports_actions=true`.
- If the artifact has no uncertainty support, sulfur upper remains unavailable
  and the history scenario can abstain.
- Reliability factors and real setpoint action candidates are still not enabled
  for the history path.
- Desktop UI exposes a forecast-only history screen; model-demo remains the only
  path that can show a synthetic recipe recommendation.

## How To Check

```bash
python -m pytest global_tests/test_stage7_history_serving.py
python -m pytest global_tests/test_stage6_acceptance.py
python -m pytest
python -m ruff check .
python -m mypy source
```

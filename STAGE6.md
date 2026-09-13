# Stage 6: acceptance and demonstration

> Current status: Stage 6 is an acceptance pack. It does not add a new ML model,
> action model, real setpoint control, or history-artifact serving in the UI.

## Purpose

Stage 6 turns the implemented prototype into a reproducible handoff package. A
second participant should be able to create an environment, run the documented
checks, reproduce the model-demo scenarios, inspect journals, and understand the
remaining limits without reading the whole codebase first.

## What Is Implemented

- `python -m source.main accept-stage6` runs the final acceptance smoke.
- The command validates configs, contracts and fixtures through `validate_stage0()`.
- It runs `blend_normal`, `blend_risk` and `blend_missing` through the existing
  deterministic model-demo cycle.
- It verifies the expected status for each scenario: all three remain `abstain`.
- It verifies that each run writes the required journal files:
  `result.json`, `input.json`, `trace.jsonl` and `candidates.jsonl`.
- It prints a JSON summary with `passed`, `validation`, `scenarios`,
  `environment`, `limitations` and `issues`.

By default, acceptance journals are written under `runs/stage6/`. For tests or
clean demos, pass an explicit temporary directory:

```bash
python -m source.main accept-stage6 --run-dir .test_tmp/stage6-runs
```

## Demonstration Flow

Recommended clean-run sequence:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m source.main validate-stage0
python -m source.main accept-stage6
python -m source.ui --smoke --scenario blend_risk
python -m pytest
```

Desktop UI remains available:

```bash
python -m source.ui --scenario blend_risk
```

The UI shows the current model-demo capability: model blending diagnostics,
constraint checks, result export, and journal inspection.

## Limits

- Stage 6 does not connect a trusted Stage-5 history artifact to CLI or UI.
- Stage 6 does not create or retrain a model.
- Stage 6 does not recommend industrial setpoint changes.
- T95, cetane number and the complete product passport remain not assessed.
- `0.95` uncertainty coverage remains an empirical historical estimate, not an
  industrial safety guarantee.

## How To Check

```bash
python -m pytest global_tests/test_stage6_acceptance.py
python -m pytest global_tests/test_ui.py global_tests/test_stage5_uncertainty_policy.py
python -m pytest
python -m ruff check .
python -m mypy source
```

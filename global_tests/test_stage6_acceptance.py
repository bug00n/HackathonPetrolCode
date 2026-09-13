"""Stage-6 acceptance command tests."""

from __future__ import annotations

import json
from pathlib import Path

from source.main import STAGE6_SCENARIOS, accept_stage6, main


def test_accept_stage6_cli_writes_journals_to_requested_run_dir(
    tmp_path: Path, capsys: object
) -> None:
    """Verify the public CLI performs the reproducible acceptance pass."""
    run_dir = tmp_path / "stage6-runs"

    exit_code = main(["accept-stage6", "--run-dir", str(run_dir)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["validation"]["model_demo_fixtures"] == 3
    assert {item["scenario_id"] for item in payload["scenarios"]} == set(STAGE6_SCENARIOS)
    for scenario in payload["scenarios"]:
        assert scenario["status"] == "abstain"
        assert scenario["passed"] is True
        journal_dir = Path(scenario["journal_dir"])
        assert journal_dir.is_relative_to(run_dir)
        for path in scenario["journal_files"].values():
            assert Path(path).is_file()


def test_accept_stage6_reports_failed_status_without_hiding_journal(
    tmp_path: Path,
) -> None:
    """Verify a status mismatch turns the acceptance result red."""
    expected = {scenario_id: "abstain" for scenario_id in STAGE6_SCENARIOS}
    expected["blend_risk"] = "hold"

    payload = accept_stage6(run_dir=tmp_path, expected_statuses=expected)

    assert payload["passed"] is False
    failed = [item for item in payload["scenarios"] if not item["passed"]]
    assert [item["scenario_id"] for item in failed] == ["blend_risk"]
    assert failed[0]["status"] == "abstain"
    assert failed[0]["expected_status"] == "hold"
    assert failed[0]["missing_journal_files"] == []
    assert any("blend_risk" in issue for issue in payload["issues"])


def test_accept_stage6_summary_shape(tmp_path: Path) -> None:
    """Verify the acceptance JSON exposes the fields used by README and demos."""
    payload = accept_stage6(run_dir=tmp_path)

    assert set(payload) == {
        "passed",
        "validation",
        "scenarios",
        "environment",
        "limitations",
        "issues",
    }
    assert isinstance(payload["validation"], dict)
    assert isinstance(payload["scenarios"], list)
    assert isinstance(payload["environment"]["python_version"], str)
    assert isinstance(payload["environment"]["sklearn_version"], str)
    assert payload["limitations"]

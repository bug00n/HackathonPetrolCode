"""Portable desktop entry point, with a hidden packaged-runtime smoke check."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from source.config import PROJECT_ROOT


def main() -> None:
    from source.main import (
        history_interval_command,
        replay_v2_shadow_command,
        run_model_demo,
        validate_stage0,
    )
    from source.ui import PetrolCodeApp
    from source.ui_data import discover_ui_context, ui_hybrid_snapshot, ui_stage_snapshot

    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-report":
        destination = Path(sys.argv[2])
        report: dict[str, object] = {}
        try:
            report["contracts"] = validate_stage0()
            report["demo"] = {
                name: run_model_demo(name).status.value
                for name in (
                    "blend_normal",
                    "blend_risk",
                    "blend_t95_risk",
                    "blend_cetane_risk",
                    "blend_missing",
                    "blend_tradeoff",
                )
            }
            context = discover_ui_context()
            at = datetime.fromisoformat("2025-06-01T12:00:00+03:00")
            dataset = context.latest_dataset or ""
            model = context.forecast_artifacts[0].path
            report["history"] = history_interval_command(
                context.latest_dataset or "",
                context.forecast_artifacts[0].path,
                datetime.fromisoformat("2025-06-01T12:00:00+03:00"),
                datetime.fromisoformat("2025-06-01T13:00:00+03:00"),
            )
            app = PetrolCodeApp(initial_page="journal")
            app.withdraw()
            app.show_page("history")
            app.update()
            app.destroy()
            report["tk"] = "ok"
            hybrid = ui_hybrid_snapshot(dataset, model, at)
            report["hybrid"] = asdict(hybrid)
            report["v2"] = replay_v2_shadow_command(dataset, context.v2_artifacts[0].path, at)
            stages = {
                page: asdict(ui_stage_snapshot(page, dataset, at))
                for page in ("avt", "hydrotreating")
            }
            report["stages"] = stages
            if hybrid.status != "ready":
                raise RuntimeError("Packaged hybrid is not ready")
            if any(view["status"] != "ready" for view in stages.values()):
                raise RuntimeError("Packaged stage snapshot is not ready")
        except Exception:
            report["error"] = traceback.format_exc()
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if "error" in report:
            raise SystemExit(1)
        return
    try:
        app = PetrolCodeApp()
        app.mainloop()
    except Exception:
        log = PROJECT_ROOT / "startup-error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        from tkinter import messagebox

        messagebox.showerror("Не удалось запустить Нефтекод", f"Подробности ошибки: {log}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

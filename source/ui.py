"""Tkinter desktop interface for model-demo and read-only research artifacts.

The UI never implies that real setpoint or product-quality action models are
available: historical effects and schema-1.2 forecasts are explicitly shadow-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Sequence

from source.config import load_scenario
from source.contracts import (
    CandidateEvaluation,
    ConstraintStatus,
    Recommendation,
    RecommendationStatus,
    ScenarioConfig,
)
from source.main import (
    PROJECT_ROOT,
    action_shadow_estimate_command,
    build_state_command,
    evaluate_lims_correction_command,
    prepare_command,
    replay_command,
    replay_v2_shadow_command,
    run_model_demo,
    validate_stage0,
)

BG = "#F5F7F7"
SURFACE = "#FFFFFF"
TEXT = "#101B28"
MUTED = "#617082"
HEADER = "#202B31"
TEAL = "#007D78"
TEAL_HOVER = "#006965"
AMBER = "#D88400"
AMBER_BG = "#FFF8E8"
BORDER = "#D8E0E4"
SOFT = "#EDF1F3"
GREEN = "#149B68"
RED = "#B3473C"

SCENARIO_LABELS = {
    "Нормальный режим": "blend_normal",
    "Повышенная сера": "blend_risk",
    "Риск T95": "blend_t95_risk",
    "Низкое цетановое число": "blend_cetane_risk",
    "Недостающие данные": "blend_missing",
}


@dataclass(frozen=True)
class ConstraintRow:
    name: str
    actual: str
    limit: str
    status: str
    tone: str


@dataclass(frozen=True)
class DashboardView:
    """Small UI projection derived exclusively from recommendation contracts."""

    status: str
    banner_title: str
    banner_detail: str
    action_title: str
    action_detail: str
    baseline_value: float | None
    baseline_t95_upper: float | None
    baseline_cetane_lower: float | None
    baseline_upper: float | None
    selected_value: float | None
    selected_t95_upper: float | None
    selected_cetane_lower: float | None
    selected_upper: float | None
    current_fractions: dict[str, float]
    proposed_fractions: dict[str, float]
    current_additive_fraction: float
    proposed_additive_fraction: float
    constraints: tuple[ConstraintRow, ...]
    explanation: str
    reasons: tuple[str, ...]


def _metric(
    evaluation: CandidateEvaluation | None, name: str
) -> tuple[float | None, float | None, float | None]:
    if evaluation is None:
        return None, None, None
    for assessment in evaluation.assessments:
        estimate = assessment.metrics.get(name)
        if estimate is not None:
            return estimate.value, estimate.lower, estimate.upper
    return None, None, None


def _limit_text(lower: float | None, upper: float | None, unit: str | None) -> str:
    suffix = f" {_display_unit(unit)}" if unit else ""
    if lower is not None and upper is not None:
        return f"{lower:g}–{upper:g}{suffix}"
    if upper is not None:
        return f"≤ {upper:g}{suffix}"
    if lower is not None:
        return f"≥ {lower:g}{suffix}"
    return "—"


def _display_unit(unit: str | None) -> str:
    return {
        "mg/kg": "мг/кг",
        "degC": "°C",
        "cetane_number": "ед.",
        "t": "т",
        "1": "доля",
    }.get(unit or "", unit or "")


def _constraint_rows(result: Recommendation) -> tuple[ConstraintRow, ...]:
    evaluation = result.selected or result.baseline
    rows: list[ConstraintRow] = []
    for check in evaluation.checks if evaluation is not None else ():
        if check.constraint_id.startswith("component_stock"):
            name = "Запас " + check.constraint_id.rsplit(":", 1)[-1]
        elif check.constraint_id == "blend_sulfur":
            name = "Сера"
        elif check.constraint_id == "blend_t95":
            name = "T95"
        elif check.constraint_id == "blend_cetane":
            name = "Цетановое число"
        elif check.constraint_id == "additive_fraction":
            name = "Доля присадки"
        elif check.constraint_id == "additive_stock":
            name = "Запас присадки"
        else:
            name = check.constraint_id
        status = {
            ConstraintStatus.PASS: "В пределах",
            ConstraintStatus.FAIL: "Нарушено",
            ConstraintStatus.UNKNOWN: "Нет оценки",
        }[check.status]
        tone = {
            ConstraintStatus.PASS: "ok",
            ConstraintStatus.FAIL: "bad",
            ConstraintStatus.UNKNOWN: "unknown",
        }[check.status]
        if check.constraint_id == "additive_fraction":
            actual = "—" if check.actual is None else f"{check.actual * 100:g} %"
            limit = _limit_text(
                None if check.lower is None else check.lower * 100,
                None if check.upper is None else check.upper * 100,
                "%",
            )
        else:
            actual = (
                "—"
                if check.actual is None
                else f"{check.actual:g} {_display_unit(check.unit)}".strip()
            )
            limit = _limit_text(check.lower, check.upper, check.unit)
        rows.append(
            ConstraintRow(
                name,
                actual,
                limit,
                status,
                tone,
            )
        )
    return tuple(rows)


def recommendation_to_view(result: Recommendation, scenario: ScenarioConfig) -> DashboardView:
    """Translate strict backend output into honest operator-facing content."""
    baseline_value, _, baseline_upper = _metric(result.baseline, "sulfur")
    _, _, baseline_t95_upper = _metric(result.baseline, "t95")
    _, baseline_cetane_lower, _ = _metric(result.baseline, "cetane_number")
    selected_value, _, selected_upper = _metric(result.selected, "sulfur")
    _, _, selected_t95_upper = _metric(result.selected, "t95")
    _, selected_cetane_lower, _ = _metric(result.selected, "cetane_number")
    current = dict(scenario.current_blend_mass_fractions)
    proposed = dict(current)
    current_additive = scenario.current_additive_mass_fraction
    proposed_additive = current_additive
    if result.selected is not None and result.selected.candidate.blend_mass_fractions:
        proposed = dict(result.selected.candidate.blend_mass_fractions)
        proposed_additive = result.selected.candidate.additive_mass_fraction

    if result.status is RecommendationStatus.RECOMMEND:
        changed = [key for key in proposed if proposed.get(key, 0.0) > current.get(key, 0.0) + 1e-9]
        component = changed[0] if changed else "смеси"
        banner_title = "Доступен модельный вариант"
        banner_detail = "Расчёт относится только к синтетическому сценарию блендинга."
        action_title = f"Увеличить долю компонента {component}"
        action_detail = "Текущая рецептура нарушает одно или несколько ограничений качества."
    elif result.status is RecommendationStatus.HOLD:
        banner_title = "Изменение режима не требуется"
        banner_detail = "Текущая модельная рецептура проходит доступные проверки."
        action_title = "Сохранить текущую рецептуру"
        action_detail = "Материального улучшения относительно hold не найдено."
    else:
        banner_title = "Рекомендация недоступна"
        banner_detail = "Обязательные данные или консервативная оценка качества отсутствуют."
        action_title = "Требуется ручная проверка"
        action_detail = "Система не подставляет неизвестные значения и не выбирает действие."

    return DashboardView(
        status=result.status.value,
        banner_title=banner_title,
        banner_detail=banner_detail,
        action_title=action_title,
        action_detail=action_detail,
        baseline_value=baseline_value,
        baseline_t95_upper=baseline_t95_upper,
        baseline_cetane_lower=baseline_cetane_lower,
        baseline_upper=baseline_upper,
        selected_value=selected_value,
        selected_t95_upper=selected_t95_upper,
        selected_cetane_lower=selected_cetane_lower,
        selected_upper=selected_upper,
        current_fractions=current,
        proposed_fractions=proposed,
        current_additive_fraction=current_additive,
        proposed_additive_fraction=proposed_additive,
        constraints=_constraint_rows(result),
        explanation=result.explanation,
        reasons=result.reason_codes,
    )


def journal_entries(run_dir: Path) -> tuple[Path, ...]:
    """Return newest journal result files first without trusting partial directories."""
    if not run_dir.exists():
        return ()
    files = [path for path in run_dir.glob("*/result.json") if path.is_file()]
    return tuple(sorted(files, key=lambda path: path.stat().st_mtime, reverse=True))


def _format_value(value: float | None, unit: str = "мг/кг") -> str:
    if value is None:
        return "—"
    rendered = f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{rendered} {unit}"


def format_action_shadow_payload(payload: dict[str, object]) -> str:
    """Render a research action scenario as sulfur changes rather than raw JSON."""
    state = payload.get("state", {})
    baseline = state.get("baseline_sulfur") if isinstance(state, dict) else None
    control = str(payload.get("control_id", "—"))
    delta = payload.get("proposed_delta")
    point = payload.get("predicted_sulfur", {})
    upper = payload.get("sulfur_upper", {})
    effect = payload.get("sulfur_change", {})
    rows = [
        "Модельный эффект по историческим эпизодам",
        "Не является советом по изменению уставки.",
        "",
        f"Текущая сера ПАК: {_format_value(baseline if isinstance(baseline, float) else None)}",
        f"Сценарий: {control} на {delta}",
        "",
        "Горизонт | Сера | Изменение к hold | Верхняя граница",
    ]
    for horizon in (60, 120, 180):
        key = str(horizon)
        predicted = point.get(key) if isinstance(point, dict) else None
        conservative = upper.get(key) if isinstance(upper, dict) else None
        change = effect.get(key) if isinstance(effect, dict) else None
        predicted_text = _format_value(predicted if isinstance(predicted, float) else None)
        change_text = _format_value(change if isinstance(change, float) else None)
        upper_text = _format_value(conservative if isinstance(conservative, float) else None)
        rows.append(f"{horizon:>3} мин | {predicted_text} | {change_text} | {upper_text}")
    rows.extend(
        (
            "",
            "Историческая область: " + ("да" if payload.get("within_observed_domain") else "нет"),
            "Validation модели: "
            + ("пройдена" if payload.get("model_validated") else "не пройдена"),
            "Верхняя граница серы: "
            + ("проходит" if payload.get("safety_passes") else "не проходит"),
        )
    )
    reasons = payload.get("reason_codes")
    if isinstance(reasons, (list, tuple)) and reasons:
        rows.append("Причины: " + ", ".join(str(reason) for reason in reasons))
    return "\n".join(rows)


def format_v2_forecast_payload(payload: dict[str, object]) -> str:
    """Render the schema-1.2 shadow forecast in an operator-readable form."""
    forecast = payload.get("forecast")
    if not isinstance(forecast, dict):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    probabilities = forecast.get("horizon_probabilities", {})
    rows = [
        "Эпизодный прогноз серы (PAK, shadow)",
        "Не является командой управления и не изменяет уставки.",
        f"Состояние на: {payload.get('as_of', '—')}",
        f"Тревога: {'да' if forecast.get('event_alarm') else 'нет'}",
        f"Применимость: {'да' if forecast.get('applicable') else 'нет'}",
        "",
        "Горизонт | P(пересечение 10 мг/кг)",
    ]
    if isinstance(probabilities, dict):
        for horizon in (10, 20, 30, 60):
            value = probabilities.get(str(horizon))
            rendered = f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "—"
            rows.append(f"{horizon:>3} мин | {rendered}")
    rows.extend(
        (
            "",
            "Точка через 60 мин: "
            + _format_value(
                forecast.get("point_60m")
                if isinstance(forecast.get("point_60m"), (int, float))
                else None
            ),
            "Верхняя граница через 60 мин: "
            + _format_value(
                forecast.get("upper_60m")
                if isinstance(forecast.get("upper_60m"), (int, float))
                else None
            ),
            "Причины: "
            + (", ".join(str(reason) for reason in forecast.get("reason_codes", ())) or "—"),
            f"Статус артефакта: {payload.get('production_status', '—')}",
        )
    )
    return "\n".join(rows)


class PetrolCodeApp(tk.Tk):
    """Resizable operator desktop shell backed by the existing Python API."""

    def __init__(
        self,
        initial_scenario: str = "blend_risk",
        initial_page: str = "overview",
    ) -> None:
        super().__init__()
        self.title("НЕФТЕКОД — поддержка технологических решений")
        self.geometry("1536x1024")
        self.minsize(1120, 760)
        self.configure(bg=BG)
        self.option_add("*Font", "{Segoe UI} 11")
        self.scenario_var = tk.StringVar(value=self._scenario_label(initial_scenario))
        self.horizon_var = tk.StringVar(value="60 мин")
        self.status_var = tk.StringVar(value="Готово к расчёту")
        self._result: Recommendation | None = None
        self._scenario: ScenarioConfig | None = None
        self._calculation_request_id = 0
        self._page = initial_page
        self._nav_buttons: dict[str, tk.Button] = {}
        self._build_styles()
        self._build_shell()
        self.calculate()

    @staticmethod
    def _scenario_label(scenario_id: str) -> str:
        return next(
            (label for label, value in SCENARIO_LABELS.items() if value == scenario_id),
            "Повышенная сера",
        )

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Petrol.Treeview",
            background=SURFACE,
            fieldbackground=SURFACE,
            foreground=TEXT,
            rowheight=38,
            borderwidth=0,
            font=("Segoe UI", 11),
        )
        style.configure(
            "Petrol.Treeview.Heading",
            background=SOFT,
            foreground="#38485A",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Petrol.Treeview", background=[("selected", "#D8EFED")])
        style.configure(
            "Petrol.TCombobox",
            fieldbackground=SURFACE,
            background=SURFACE,
            foreground=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=9,
        )

    def _build_shell(self) -> None:
        self.header = tk.Frame(self, bg=HEADER, height=60)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)
        tk.Label(
            self.header,
            text="НЕФТЕКОД",
            bg=HEADER,
            fg="white",
            font=("Segoe UI", 20, "bold"),
        ).pack(side="left", padx=(32, 28))
        tk.Frame(self.header, bg="#65727A", width=1, height=26).pack(side="left", padx=(0, 14))
        for key, label in (
            ("overview", "Обзор"),
            ("recommendation", "Рекомендации"),
            ("journal", "Журнал"),
        ):
            button = tk.Button(
                self.header,
                text=label,
                command=partial(self.show_page, key),
                bg=HEADER,
                fg="#C0C8CD",
                activebackground=HEADER,
                activeforeground="white",
                bd=0,
                padx=18,
                font=("Segoe UI", 11, "bold"),
                cursor="hand2",
            )
            button.pack(side="left", fill="y")
            self._nav_buttons[key] = button
        tk.Label(
            self.header,
            text="Модельный контур  •  реальные уставки отключены",
            bg=HEADER,
            fg="#B8C3C9",
            font=("Segoe UI", 10),
        ).pack(side="right", padx=32)
        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True)
        self.show_page(self._page)

    def _clear_body(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()

    def _set_nav(self, page: str) -> None:
        for key, button in self._nav_buttons.items():
            button.configure(fg="white" if key == page else "#C0C8CD")

    def show_page(self, page: str) -> None:
        self._page = page
        self._set_nav(page)
        self._clear_body()
        if page == "overview":
            self._render_overview()
        elif page == "recommendation":
            self._render_recommendation()
        else:
            self._render_journal()

    def _page_container(self) -> tk.Frame:
        canvas = tk.Canvas(self.body, bg=BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        frame = tk.Frame(canvas, bg=BG)
        window_id = canvas.create_window(0, 0, anchor="nw", window=frame)

        def resize_page(event: tk.Event[tk.Misc]) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def update_scroll_region(_: tk.Event[tk.Misc]) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def scroll_page(event: tk.Event[tk.Misc]) -> None:
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind("<Configure>", resize_page)
        self.bind_all("<MouseWheel>", scroll_page)
        frame.bind("<Configure>", update_scroll_region)
        frame.configure(padx=32, pady=24)
        return frame

    @staticmethod
    def _surface(parent: tk.Misc, **grid: Any) -> tk.Frame:
        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        if grid:
            frame.grid(**grid)
        return frame

    @staticmethod
    def _primary_button(parent: tk.Misc, text: str, command: Any) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=TEAL,
            fg="white",
            activebackground=TEAL_HOVER,
            activeforeground="white",
            bd=0,
            padx=24,
            pady=11,
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        )

    @staticmethod
    def _secondary_button(parent: tk.Misc, text: str, command: Any) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=SURFACE,
            fg=TEXT,
            activebackground=SOFT,
            activeforeground=TEXT,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
            padx=20,
            pady=10,
            cursor="hand2",
        )

    def _render_overview(self) -> None:
        page = self._page_container()
        top = tk.Frame(page, bg=BG)
        top.pack(fill="x")
        title = tk.Frame(top, bg=BG)
        title.pack(side="left")
        tk.Label(
            title,
            text="Состояние установки",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 28, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title,
            text="Текущий модельный сценарий, расчёт и ограничения",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 12),
        ).pack(anchor="w")
        controls = tk.Frame(top, bg=BG)
        controls.pack(side="right", pady=8)
        scenario = ttk.Combobox(
            controls,
            values=tuple(SCENARIO_LABELS),
            textvariable=self.scenario_var,
            state="readonly",
            width=23,
            style="Petrol.TCombobox",
        )
        scenario.pack(side="left", padx=6)
        horizon = ttk.Combobox(
            controls,
            values=("60 мин",),
            textvariable=self.horizon_var,
            state="readonly",
            width=12,
            style="Petrol.TCombobox",
        )
        horizon.pack(side="left", padx=6)
        button = self._primary_button(controls, "Рассчитать", self.calculate)
        button.pack(side="left", padx=(8, 0))

        view = self._view()
        banner = tk.Frame(
            page,
            bg=AMBER_BG,
            highlightbackground="#E9B850",
            highlightthickness=1,
        )
        banner.pack(fill="x", pady=(22, 16), ipady=12)
        tk.Label(
            banner,
            text="⚠",
            bg=AMBER_BG,
            fg=AMBER,
            font=("Segoe UI Symbol", 23, "bold"),
        ).pack(side="left", padx=(20, 14))
        banner_text = tk.Frame(banner, bg=AMBER_BG)
        banner_text.pack(side="left")
        tk.Label(
            banner_text,
            text=view.banner_title if view else self.status_var.get(),
            bg=AMBER_BG,
            fg="#583714",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            banner_text,
            text=view.banner_detail if view else "Расчёт ещё не выполнен.",
            bg=AMBER_BG,
            fg="#536171",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        main = self._surface(page)
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=4)
        main.grid_columnconfigure(1, weight=1, minsize=260)
        main.grid_rowconfigure(0, weight=1)
        chart_side = tk.Frame(main, bg=SURFACE)
        chart_side.grid(row=0, column=0, sticky="nsew")
        info = tk.Frame(main, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        info.grid(row=0, column=1, sticky="nsew")
        self._render_kpis(chart_side, view)
        self._render_chart(chart_side, view)
        self._render_info(info, view)

        stages = tk.Frame(page, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        stages.pack(fill="x", pady=(16, 12), ipady=4)
        for text, enabled in (("АВТ", False), ("Гидроочистка", False), ("Смесь", True)):
            command = (
                partial(self.show_page, "recommendation")
                if enabled
                else self._show_stage_placeholder
            )
            button = tk.Button(
                stages,
                text=text,
                command=command,
                bg=TEAL if enabled else SOFT,
                fg="white" if enabled else TEXT,
                activebackground=TEAL_HOVER if enabled else SOFT,
                bd=0,
                pady=8,
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
            )
            button.pack(side="left", fill="x", expand=True, padx=3)
        self._accordion(
            page,
            "Источники, проверка конфигурации и подготовка данных",
            self._data_tools_content,
        ).pack(fill="x")

    def _render_kpis(self, parent: tk.Frame, view: DashboardView | None) -> None:
        strip = tk.Frame(parent, bg=SURFACE)
        strip.pack(fill="x", padx=26, pady=(18, 12))
        values = (
            ("Сера · верхняя", _format_value(view.selected_upper) if view else "—"),
            (
                "T95 · верхняя",
                _format_value(view.selected_t95_upper, "°C") if view else "—",
            ),
            (
                "Цетановое · нижняя",
                _format_value(view.selected_cetane_lower, "ед.") if view else "—",
            ),
        )
        for index, (label, value) in enumerate(values):
            cell = tk.Frame(strip, bg=SURFACE)
            cell.pack(side="left", fill="x", expand=True, padx=(0, 18))
            tk.Label(cell, text=label, bg=SURFACE, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(
                anchor="w"
            )
            tk.Label(cell, text=value, bg=SURFACE, fg=TEXT, font=("Segoe UI", 25, "bold")).pack(
                anchor="w", pady=(3, 0)
            )
            tk.Label(
                cell, text="Модельный расчёт", bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)
            ).pack(anchor="w")
            if index < 2:
                tk.Frame(strip, bg=BORDER, width=1, height=72).pack(side="left", padx=(0, 18))

    def _render_chart(self, parent: tk.Frame, view: DashboardView | None) -> None:
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=24)
        tk.Label(
            parent,
            text="Сера в смеси, мг/кг",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=24, pady=(16, 2))
        canvas = tk.Canvas(parent, bg=SURFACE, highlightthickness=0, height=345)
        canvas.pack(fill="both", expand=True, padx=24, pady=(0, 14))

        def redraw(_: tk.Event[Any] | None = None) -> None:
            self._draw_sulfur_chart(canvas, view)

        canvas.bind("<Configure>", redraw)
        redraw()

    @staticmethod
    def _draw_sulfur_chart(canvas: tk.Canvas, view: DashboardView | None) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 680)
        height = max(canvas.winfo_height(), 300)
        left, right, top, bottom = 52, width - 26, 34, height - 48
        maximum = 16.0
        for value in range(0, 17, 2):
            y = bottom - (value / maximum) * (bottom - top)
            canvas.create_line(left, y, right, y, fill="#E3E8EA")
            canvas.create_text(left - 12, y, text=str(value), fill=MUTED, anchor="e")
        canvas.create_line(left, bottom, right, bottom, fill="#AEBAC1")
        limit_y = bottom - (10.0 / maximum) * (bottom - top)
        canvas.create_line(left, limit_y, right, limit_y, fill=AMBER, dash=(7, 5), width=2)
        canvas.create_text(
            right - 6,
            limit_y - 12,
            text="Предел сценария · 10",
            fill=AMBER,
            anchor="e",
            font=("Segoe UI", 9, "bold"),
        )
        x_current = left + (right - left) * 0.28
        x_selected = left + (right - left) * 0.72
        baseline = view.baseline_upper if view and view.baseline_upper is not None else None
        selected = view.selected_upper if view and view.selected_upper is not None else None
        if baseline is not None:
            y_current = bottom - min(baseline, maximum) / maximum * (bottom - top)
            canvas.create_oval(
                x_current - 6, y_current - 6, x_current + 6, y_current + 6, fill=TEAL, outline=""
            )
            canvas.create_text(
                x_current,
                y_current - 18,
                text=_format_value(baseline),
                fill=TEXT,
                font=("Segoe UI", 10, "bold"),
            )
            if selected is not None:
                y_selected = bottom - min(selected, maximum) / maximum * (bottom - top)
                canvas.create_line(
                    x_current, y_current, x_selected, y_selected, fill=TEAL, dash=(7, 5), width=3
                )
                canvas.create_oval(
                    x_selected - 6,
                    y_selected - 6,
                    x_selected + 6,
                    y_selected + 6,
                    fill=TEAL,
                    outline="",
                )
                canvas.create_text(
                    x_selected,
                    y_selected - 18,
                    text=_format_value(selected),
                    fill=TEXT,
                    font=("Segoe UI", 10, "bold"),
                )
        else:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Расчёт серы недоступен",
                fill=MUTED,
                font=("Segoe UI", 13),
            )
        canvas.create_text(x_current, bottom + 24, text="Текущая рецептура", fill=MUTED)
        canvas.create_text(x_selected, bottom + 24, text="Выбранный вариант", fill=MUTED)
        canvas.create_text(
            right,
            top - 14,
            text="Показана верхняя сценарная оценка, не исторический тренд",
            fill=MUTED,
            anchor="e",
            font=("Segoe UI", 9),
        )

    def _render_info(self, parent: tk.Frame, view: DashboardView | None) -> None:
        tk.Label(parent, text="Данные", bg=SURFACE, fg=TEXT, font=("Segoe UI", 15, "bold")).pack(
            anchor="w", padx=22, pady=(20, 12)
        )
        self._info_row(parent, "Режим", "Модельный")
        self._info_row(parent, "Сценарий", SCENARIO_LABELS.get(self.scenario_var.get(), "—"))
        self._info_row(parent, "Горизонт", self.horizon_var.get())
        self._info_row(parent, "Последний расчёт", datetime.now().strftime("%H:%M"))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=18)
        tk.Label(
            parent,
            text="Надёжность: нет промышленной оценки",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 10, "bold"),
            wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=22)
        tk.Label(
            parent,
            text=(
                "Action model не подтверждён. Реальные управляющие воздействия отключены."
                if view
                else "Выполните расчёт."
            ),
            bg=SURFACE,
            fg=MUTED,
            wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=22, pady=(8, 0))

    @staticmethod
    def _info_row(parent: tk.Frame, label: str, value: str) -> None:
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", padx=22, pady=6)
        tk.Label(row, text=label, bg=SURFACE, fg=MUTED).pack(side="left")
        tk.Label(row, text=value, bg=SURFACE, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(
            side="right"
        )

    def _render_recommendation(self) -> None:
        page = self._page_container()
        view = self._view()
        heading = tk.Frame(page, bg=BG)
        heading.pack(fill="x")
        tk.Label(
            heading,
            text="Рекомендация по смешению",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 28, "bold"),
        ).pack(side="left")
        badge = tk.Label(
            heading,
            text="⚠  Модельный режим",
            bg=AMBER_BG,
            fg="#A55E00",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
            highlightbackground=AMBER,
            highlightthickness=1,
        )
        badge.pack(side="left", padx=22)
        tk.Label(
            page,
            text="Результат применим только к модельному сценарию; реальные уставки не меняются",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(2, 18))

        main = self._surface(page)
        main.pack(fill="x")
        main.grid_columnconfigure(0, weight=2)
        main.grid_columnconfigure(1, weight=1)
        left = tk.Frame(main, bg=SURFACE)
        left.grid(row=0, column=0, sticky="nsew", padx=26, pady=22)
        right = tk.Frame(main, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=14)
        tk.Label(
            left,
            text="РЕКОМЕНДАЦИЯ",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text=view.action_title if view else "Расчёт не выполнен",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", pady=(8, 2))
        tk.Label(
            left,
            text=view.action_detail if view else "Вернитесь на экран обзора.",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 12),
        ).pack(anchor="w", pady=(0, 14))
        self._recipe_table(left, view)
        tk.Label(
            left,
            text="Массовые доли · сумма 100 % · значения синтетические",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(8, 0))
        tk.Label(
            right,
            text="ОЖИДАЕМЫЙ РЕЗУЛЬТАТ",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=20, pady=(18, 0))
        tk.Label(
            right,
            text=_format_value(view.selected_upper if view else None),
            bg=SURFACE,
            fg=TEAL if view and view.selected_upper is not None else MUTED,
            font=("Segoe UI", 30, "bold"),
        ).pack(anchor="w", padx=20, pady=(8, 0))
        tk.Label(right, text="Верхняя оценка серы смеси", bg=SURFACE, fg=MUTED).pack(
            anchor="w", padx=20
        )
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=20, pady=18)
        self._info_row(right, "Текущая", _format_value(view.baseline_upper if view else None))
        self._info_row(right, "Предел сценария", "≤ 10 мг/кг")
        self._info_row(
            right,
            "T95 · верхняя",
            _format_value(view.selected_t95_upper if view else None, "°C"),
        )
        self._info_row(
            right,
            "Цетановое · нижняя",
            _format_value(view.selected_cetane_lower if view else None, "ед."),
        )

        checks = self._surface(page)
        checks.pack(fill="x", pady=(16, 0))
        tk.Label(
            checks,
            text="Проверка ограничений",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", padx=26, pady=(14, 8))
        self._constraint_table(checks, view.constraints if view else ())
        tk.Label(
            checks,
            text="Система подтверждает только строки с рассчитанным значением и источником.",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=26, pady=(7, 12))

        self._accordion(
            page,
            "Почему выбран этот вариант",
            lambda parent: self._text_content(parent, view.explanation if view else "Нет расчёта"),
        ).pack(fill="x", pady=(16, 8))
        self._accordion(
            page,
            "Альтернативы и ход расчёта",
            self._alternatives_content,
        ).pack(fill="x")
        footer = tk.Frame(page, bg=BG)
        footer.pack(fill="x", pady=(18, 0))
        self._primary_button(footer, "Скачать расчёт", self.export_result).pack(side="left")
        self._secondary_button(footer, "Открыть журнал", lambda: self.show_page("journal")).pack(
            side="left", padx=14
        )
        tk.Label(
            footer,
            text="НЕФТЕКОД  v1.0  |  prototype",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="right", pady=12)

    def _recipe_table(self, parent: tk.Frame, view: DashboardView | None) -> None:
        table = ttk.Treeview(
            parent,
            columns=("component", "current", "proposed"),
            show="headings",
            height=max(3, len(view.current_fractions) + 1 if view else 3),
            style="Petrol.Treeview",
        )
        for key, title, width in (
            ("component", "Компонент", 220),
            ("current", "Сейчас", 150),
            ("proposed", "Предлагается", 170),
        ):
            table.heading(key, text=title)
            table.column(key, width=width, anchor="w", stretch=True)
        if view:
            for component in sorted(view.current_fractions):
                table.insert(
                    "",
                    "end",
                    values=(
                        f"Компонент {component}",
                        f"{view.current_fractions[component] * 100:.0f} %",
                        f"{view.proposed_fractions.get(component, 0.0) * 100:.0f} %",
                    ),
                )
            table.insert(
                "",
                "end",
                values=(
                    "Цетаноповышающая присадка",
                    f"{view.current_additive_fraction * 100:.1f} %",
                    f"{view.proposed_additive_fraction * 100:.1f} %",
                ),
            )
        table.pack(fill="x")

    @staticmethod
    def _constraint_table(parent: tk.Frame, rows: tuple[ConstraintRow, ...]) -> None:
        table = ttk.Treeview(
            parent,
            columns=("name", "actual", "limit", "status"),
            show="headings",
            height=max(4, len(rows)),
            style="Petrol.Treeview",
        )
        for key, title, width in (
            ("name", "Показатель", 240),
            ("actual", "Расчёт", 210),
            ("limit", "Предел сценария", 250),
            ("status", "Статус", 210),
        ):
            table.heading(key, text=title)
            table.column(key, width=width, anchor="w", stretch=True)
        table.tag_configure("ok", foreground=GREEN)
        table.tag_configure("bad", foreground=RED)
        table.tag_configure("unknown", foreground=AMBER)
        for row in rows:
            table.insert(
                "", "end", values=(row.name, row.actual, row.limit, row.status), tags=(row.tone,)
            )
        table.pack(fill="x", padx=26)

    def _render_journal(self) -> None:
        page = self._page_container()
        tk.Label(
            page,
            text="Журнал расчётов",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 28, "bold"),
        ).pack(anchor="w")
        tk.Label(
            page,
            text="Сохранённые backend-результаты; записи не редактируются из интерфейса",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(2, 18))
        content = self._surface(page)
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        paths = journal_entries(PROJECT_ROOT / "runs")
        listbox = tk.Listbox(
            content,
            bg=SURFACE,
            fg=TEXT,
            selectbackground="#D8EFED",
            selectforeground=TEXT,
            bd=0,
            highlightthickness=0,
            width=38,
            font=("Consolas", 10),
        )
        listbox.grid(row=0, column=0, sticky="nsew", padx=(18, 0), pady=18)
        detail = tk.Text(
            content,
            bg="#FAFBFB",
            fg=TEXT,
            bd=0,
            padx=18,
            pady=14,
            wrap="word",
            font=("Consolas", 10),
        )
        detail.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        for path in paths:
            listbox.insert("end", path.parent.name)

        def select(_: tk.Event[Any] | None = None) -> None:
            selection = listbox.curselection()
            if not selection:
                return
            payload = json.loads(paths[selection[0]].read_text(encoding="utf-8"))
            detail.configure(state="normal")
            detail.delete("1.0", "end")
            detail.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))
            detail.configure(state="disabled")

        listbox.bind("<<ListboxSelect>>", select)
        if paths:
            listbox.selection_set(0)
            select()
        else:
            detail.insert("1.0", "Журнал пуст. Выполните модельный расчёт на экране «Обзор».")
            detail.configure(state="disabled")

    def _accordion(self, parent: tk.Misc, title: str, builder: Any) -> tk.Frame:
        shell = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        content = tk.Frame(shell, bg=SURFACE)
        opened = tk.BooleanVar(value=False)

        def toggle() -> None:
            if opened.get():
                content.pack_forget()
                opened.set(False)
                button.configure(text=f"›  {title}")
            else:
                if not content.winfo_children():
                    builder(content)
                content.pack(fill="x", padx=24, pady=(0, 16))
                opened.set(True)
                button.configure(text=f"⌄  {title}")

        button = tk.Button(
            shell,
            text=f"›  {title}",
            command=toggle,
            anchor="w",
            bg=SURFACE,
            fg=TEXT,
            activebackground=SOFT,
            bd=0,
            padx=18,
            pady=12,
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        )
        button.pack(fill="x")
        return shell

    @staticmethod
    def _text_content(parent: tk.Frame, text: str) -> None:
        tk.Label(
            parent,
            text=text,
            bg=SURFACE,
            fg=MUTED,
            wraplength=1250,
            justify="left",
        ).pack(anchor="w")

    def _alternatives_content(self, parent: tk.Frame) -> None:
        if self._result is None or not self._result.alternatives:
            self._text_content(parent, "Допустимых альтернатив нет.")
            return
        lines = []
        for alternative in self._result.alternatives:
            value, _, upper = _metric(alternative, "sulfur")
            lines.append(
                f"{alternative.candidate.id}: point {_format_value(value)}, "
                f"upper {_format_value(upper)}"
            )
        self._text_content(parent, "\n".join(lines))

    def _data_tools_content(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text=(
                "Доступны те же безопасные операции, что и в CLI. Подготовка полного архива "
                "может занять несколько минут."
            ),
            bg=SURFACE,
            fg=MUTED,
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x")
        self._secondary_button(row, "Проверить конфигурацию", self.validate_config).pack(
            side="left"
        )
        self._secondary_button(row, "Подготовить данные", self.prepare_data).pack(
            side="left", padx=10
        )
        self._secondary_button(row, "Собрать historical state", self.open_state_dialog).pack(
            side="left"
        )
        self._secondary_button(row, "Прогноз серы", self.open_history_forecast_dialog).pack(
            side="left", padx=(10, 0)
        )
        self._secondary_button(row, "Эпизодный прогноз v2", self.open_v2_forecast_dialog).pack(
            side="left", padx=(10, 0)
        )
        self._secondary_button(row, "Контроль ПАК–ЛИМС", self.open_lims_correction_dialog).pack(
            side="left", padx=(10, 0)
        )
        self._secondary_button(row, "Исследовать P8/F19", self.open_action_shadow_dialog).pack(
            side="left", padx=(10, 0)
        )

    def _view(self) -> DashboardView | None:
        if self._result is None or self._scenario is None:
            return None
        return recommendation_to_view(self._result, self._scenario)

    def calculate(self) -> None:
        self._calculation_request_id += 1
        request_id = self._calculation_request_id
        scenario_id = SCENARIO_LABELS[self.scenario_var.get()]
        self.status_var.set("Выполняется расчёт…")

        def work() -> None:
            try:
                scenario = load_scenario(PROJECT_ROOT / f"config/scenarios/{scenario_id}.json")
                result = run_model_demo(scenario_id)
            except Exception as exc:  # UI boundary: render backend failure without crashing Tk.
                self.after(0, partial(self._calculation_failed, request_id, exc))
                return
            self.after(0, lambda: self._calculation_done(request_id, result, scenario))

        threading.Thread(target=work, daemon=True).start()

    def _calculation_done(
        self, request_id: int, result: Recommendation, scenario: ScenarioConfig
    ) -> None:
        if request_id != self._calculation_request_id:
            return
        self._result = result
        self._scenario = scenario
        self.status_var.set("Расчёт завершён")
        self.show_page(self._page)

    def _calculation_failed(self, request_id: int, error: Exception) -> None:
        if request_id != self._calculation_request_id:
            return
        self.status_var.set("Ошибка расчёта")
        messagebox.showerror("Ошибка расчёта", str(error), parent=self)
        self.show_page(self._page)

    def export_result(self) -> None:
        if self._result is None:
            messagebox.showinfo("Нет расчёта", "Сначала выполните расчёт.", parent=self)
            return
        destination = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить расчёт",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"),),
            initialfile=f"neftekod-{self._result.scenario_id}-{self._result.run_id[:8]}.json",
        )
        if destination:
            Path(destination).write_text(
                self._result.model_dump_json(indent=2), encoding="utf-8", newline="\n"
            )

    def validate_config(self) -> None:
        try:
            result = validate_stage0()
        except Exception as exc:
            messagebox.showerror("Проверка не пройдена", str(exc), parent=self)
            return
        messagebox.showinfo(
            "Конфигурация корректна",
            f"Теги: {result['tags']}\nСценарии: {result['scenarios']}\n"
            f"Demo fixtures: {result['model_demo_fixtures']}",
            parent=self,
        )

    def prepare_data(self) -> None:
        if not messagebox.askokcancel(
            "Подготовить данные",
            "Прочитать полный архив materials и создать локальный prepared dataset?",
            parent=self,
        ):
            return
        self.status_var.set("Подготовка данных…")

        def work() -> None:
            try:
                result = prepare_command("materials", "config/runtime.toml")
            except Exception as exc:
                self.after(0, partial(self._preparation_failed, exc))
                return
            self.after(0, partial(self._preparation_done, result))

        threading.Thread(target=work, daemon=True).start()

    def _preparation_done(self, result: dict[str, Any]) -> None:
        self.status_var.set("Данные подготовлены")
        messagebox.showinfo(
            "Данные подготовлены",
            json.dumps(result, ensure_ascii=False, indent=2),
            parent=self,
        )

    def _preparation_failed(self, error: Exception) -> None:
        self.status_var.set("Ошибка подготовки данных")
        messagebox.showerror("Ошибка подготовки", str(error), parent=self)

    def open_state_dialog(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Собрать historical state")
        dialog.geometry("680x330")
        dialog.configure(bg=BG)
        dialog.transient(self)
        fields = tk.Frame(dialog, bg=BG)
        fields.pack(fill="both", expand=True, padx=26, pady=22)
        tk.Label(fields, text="Prepared dataset", bg=BG, fg=TEXT).pack(anchor="w")
        datasets = sorted((PROJECT_ROOT / "data/processed").glob("*/manifest.json"))
        dataset_var = tk.StringVar(value=str(datasets[-1].parent) if datasets else "")
        tk.Entry(fields, textvariable=dataset_var, bg=SURFACE, fg=TEXT, bd=1).pack(
            fill="x", pady=(4, 14), ipady=7
        )
        tk.Label(fields, text="As of (ISO с timezone)", bg=BG, fg=TEXT).pack(anchor="w")
        as_of_var = tk.StringVar(value="2025-01-15T10:00:00+03:00")
        tk.Entry(fields, textvariable=as_of_var, bg=SURFACE, fg=TEXT, bd=1).pack(
            fill="x", pady=(4, 14), ipady=7
        )
        output = tk.Text(fields, height=5, bg=SURFACE, fg=TEXT, bd=1, wrap="word")
        output.pack(fill="both", expand=True)

        def build() -> None:
            try:
                timestamp = datetime.fromisoformat(as_of_var.get().replace("Z", "+00:00"))
                state = build_state_command(dataset_var.get(), "history", timestamp)
                text = state.model_dump_json(indent=2)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            output.delete("1.0", "end")
            output.insert("1.0", text)

        self._primary_button(fields, "Собрать состояние", build).pack(anchor="e", pady=(12, 0))

    def open_action_shadow_dialog(self) -> None:
        """Show a historical P8/F19 scenario without presenting it as a recommendation."""
        dialog = tk.Toplevel(self)
        dialog.title("Модельный эффект по историческим эпизодам")
        dialog.geometry("760x620")
        dialog.configure(bg=BG)
        dialog.transient(self)
        fields = tk.Frame(dialog, bg=BG)
        fields.pack(fill="both", expand=True, padx=26, pady=22)
        datasets = sorted((PROJECT_ROOT / "data/processed").glob("*/manifest.json"))
        artifacts = sorted(
            (PROJECT_ROOT / "artifacts/models").glob("action-shadow-*/metadata.json"),
            key=lambda path: path.stat().st_mtime,
        )
        dataset_var = tk.StringVar(value=str(datasets[-1].parent) if datasets else "")
        model_var = tk.StringVar(value=str(artifacts[-1].parent) if artifacts else "")
        control_var = tk.StringVar(value="ht:P8")
        delta_var = tk.StringVar(value="0.001")
        at_var = tk.StringVar(value="2025-06-01T12:00:00+03:00")
        for label, variable in (
            ("Prepared dataset", dataset_var),
            ("Action shadow artifact", model_var),
            ("Время состояния (ISO с timezone)", at_var),
            ("Изменение тега в его исходной единице", delta_var),
        ):
            tk.Label(fields, text=label, bg=BG, fg=TEXT).pack(anchor="w")
            tk.Entry(fields, textvariable=variable, bg=SURFACE, fg=TEXT, bd=1).pack(
                fill="x", pady=(4, 10), ipady=6
            )
        tk.Label(fields, text="Тег", bg=BG, fg=TEXT).pack(anchor="w")
        ttk.Combobox(
            fields,
            textvariable=control_var,
            values=("ht:P8", "ht:F19"),
            state="readonly",
            style="Petrol.TCombobox",
        ).pack(fill="x", pady=(4, 10))
        tk.Label(
            fields,
            text=(
                "Расчёт показывает модельный эффект в истории. Он не является советом "
                "по изменению уставки и не включает управление оборудованием."
            ),
            bg=BG,
            fg=MUTED,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        output = tk.Text(fields, height=14, bg=SURFACE, fg=TEXT, bd=1, wrap="word")
        output.pack(fill="both", expand=True)

        def render(text: str) -> None:
            output.delete("1.0", "end")
            output.insert("1.0", text)

        def calculate() -> None:
            try:
                at = datetime.fromisoformat(at_var.get().replace("Z", "+00:00"))
                if at.tzinfo is None:
                    raise ValueError("время должно содержать timezone")
                result = action_shadow_estimate_command(
                    dataset_var.get(),
                    model_var.get(),
                    control_var.get(),
                    float(delta_var.get()),
                    at,
                )
                text = format_action_shadow_payload(result)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self.after(0, lambda: render(text))

        self._primary_button(
            fields,
            "Рассчитать исследовательский сценарий",
            lambda: threading.Thread(target=calculate, daemon=True).start(),
        ).pack(anchor="e", pady=(12, 0))

    def open_history_forecast_dialog(self) -> None:
        """Replay a trusted local sulfur artifact at one historical timestamp."""
        dialog = tk.Toplevel(self)
        dialog.title("Исторический прогноз серы")
        dialog.geometry("760x570")
        dialog.configure(bg=BG)
        dialog.transient(self)
        fields = tk.Frame(dialog, bg=BG)
        fields.pack(fill="both", expand=True, padx=26, pady=22)
        datasets = sorted((PROJECT_ROOT / "data/processed").glob("*/manifest.json"))
        artifacts = sorted((PROJECT_ROOT / "artifacts/models").glob("*/metadata.json"))
        forecast_artifacts = [
            path for path in artifacts if not path.parent.name.startswith("action-shadow-")
        ]
        forecast_artifacts.sort(key=lambda path: path.stat().st_mtime)
        dataset_var = tk.StringVar(value=str(datasets[-1].parent) if datasets else "")
        model_var = tk.StringVar(
            value=str(forecast_artifacts[-1].parent) if forecast_artifacts else ""
        )
        at_var = tk.StringVar(value="2025-06-01T12:00:00+03:00")
        for label, variable in (
            ("Prepared dataset", dataset_var),
            ("Forecast artifact", model_var),
            ("Время состояния (ISO с timezone)", at_var),
        ):
            tk.Label(fields, text=label, bg=BG, fg=TEXT).pack(anchor="w")
            tk.Entry(fields, textvariable=variable, bg=SURFACE, fg=TEXT, bd=1).pack(
                fill="x", pady=(4, 12), ipady=6
            )
        tk.Label(
            fields,
            text=(
                "Прогноз использует только доступные к выбранному времени данные. "
                "Он предупреждает о риске качества, но не изменяет уставки."
            ),
            bg=BG,
            fg=MUTED,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        output = tk.Text(fields, height=13, bg=SURFACE, fg=TEXT, bd=1, wrap="word")
        output.pack(fill="both", expand=True)

        def render(text: str) -> None:
            output.delete("1.0", "end")
            output.insert("1.0", text)

        def calculate() -> None:
            try:
                at = datetime.fromisoformat(at_var.get().replace("Z", "+00:00"))
                if at.tzinfo is None:
                    raise ValueError("время должно содержать timezone")
                result = replay_command(dataset_var.get(), model_var.get(), "history", at)
                text = result.model_dump_json(indent=2)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self.after(0, lambda: render(text))

        self._primary_button(
            fields,
            "Рассчитать прогноз",
            lambda: threading.Thread(target=calculate, daemon=True).start(),
        ).pack(anchor="e", pady=(12, 0))

    def open_v2_forecast_dialog(self) -> None:
        """Serve the multi-horizon schema-1.2 artifact as a shadow forecast."""
        dialog = tk.Toplevel(self)
        dialog.title("Эпизодный прогноз серы v2")
        dialog.geometry("760x570")
        dialog.configure(bg=BG)
        dialog.transient(self)
        fields = tk.Frame(dialog, bg=BG)
        fields.pack(fill="both", expand=True, padx=26, pady=22)
        datasets = sorted((PROJECT_ROOT / "data/processed").glob("*/manifest.json"))
        v2_artifacts: list[Path] = []
        for metadata_path in sorted((PROJECT_ROOT / "artifacts/models").glob("*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            capabilities = metadata.get("capabilities")
            if (
                metadata.get("schema_version") == "1.2"
                and isinstance(capabilities, dict)
                and capabilities.get("supports_multi_horizon")
            ):
                v2_artifacts.append(metadata_path)
        dataset_var = tk.StringVar(value=str(datasets[-1].parent) if datasets else "")
        v2_artifacts.sort(key=lambda path: path.stat().st_mtime)
        model_var = tk.StringVar(value=str(v2_artifacts[-1].parent) if v2_artifacts else "")
        at_var = tk.StringVar(value="2025-06-01T12:00:00+03:00")
        for label, variable in (
            ("Prepared dataset", dataset_var),
            ("Schema-1.2 shadow artifact", model_var),
            ("Время состояния (ISO с timezone)", at_var),
        ):
            tk.Label(fields, text=label, bg=BG, fg=TEXT).pack(anchor="w")
            tk.Entry(fields, textvariable=variable, bg=SURFACE, fg=TEXT, bd=1).pack(
                fill="x", pady=(4, 12), ipady=6
            )
        tk.Label(
            fields,
            text=(
                "Показывает вероятность начала эпизода и верхнюю границу серы на нескольких "
                "горизонтах. Артефакт работает только в shadow-режиме: это предупреждение, "
                "а не совет и не изменение уставок."
            ),
            bg=BG,
            fg=MUTED,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        output = tk.Text(fields, height=13, bg=SURFACE, fg=TEXT, bd=1, wrap="word")
        output.pack(fill="both", expand=True)

        def render(text: str) -> None:
            output.delete("1.0", "end")
            output.insert("1.0", text)

        def calculate() -> None:
            try:
                at = datetime.fromisoformat(at_var.get().replace("Z", "+00:00"))
                if at.tzinfo is None:
                    raise ValueError("время должно содержать timezone")
                result = replay_v2_shadow_command(dataset_var.get(), model_var.get(), at)
                text = format_v2_forecast_payload(result)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self.after(0, lambda: render(text))

        self._primary_button(
            fields,
            "Рассчитать episode forecast",
            lambda: threading.Thread(target=calculate, daemon=True).start(),
        ).pack(anchor="e", pady=(12, 0))

    def open_lims_correction_dialog(self) -> None:
        """Evaluate the separately labelled delayed LIMS correction evidence."""
        dialog = tk.Toplevel(self)
        dialog.title("Контроль ПАК–ЛИМС")
        dialog.geometry("760x500")
        dialog.configure(bg=BG)
        dialog.transient(self)
        fields = tk.Frame(dialog, bg=BG)
        fields.pack(fill="both", expand=True, padx=26, pady=22)
        datasets = sorted((PROJECT_ROOT / "data/processed").glob("*/manifest.json"))
        artifacts = sorted((PROJECT_ROOT / "artifacts/models").glob("*/metadata.json"))
        forecast_artifacts = [
            path for path in artifacts if not path.parent.name.startswith("action-shadow-")
        ]
        forecast_artifacts.sort(key=lambda path: path.stat().st_mtime)
        dataset_var = tk.StringVar(value=str(datasets[-1].parent) if datasets else "")
        model_var = tk.StringVar(
            value=str(forecast_artifacts[-1].parent) if forecast_artifacts else ""
        )
        for label, variable in (
            ("Prepared dataset", dataset_var),
            ("PAK forecast artifact", model_var),
        ):
            tk.Label(fields, text=label, bg=BG, fg=TEXT).pack(anchor="w")
            tk.Entry(fields, textvariable=variable, bg=SURFACE, fg=TEXT, bd=1).pack(
                fill="x", pady=(4, 12), ipady=6
            )
        tk.Label(
            fields,
            text=(
                "ЛИМС публикуется с задержкой и используется отдельно от оперативного ПАК. "
                "Результат показывает качество коррекции; только прошедшая gate-коррекция "
                "может стать кандидатом для shadow-периода."
            ),
            bg=BG,
            fg=MUTED,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        output = tk.Text(fields, height=14, bg=SURFACE, fg=TEXT, bd=1, wrap="word")
        output.pack(fill="both", expand=True)

        def render(text: str) -> None:
            output.delete("1.0", "end")
            output.insert("1.0", text)

        def calculate() -> None:
            try:
                result = evaluate_lims_correction_command(dataset_var.get(), model_var.get())
                text = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self.after(0, lambda: render(text))

        self._primary_button(
            fields,
            "Оценить LIMS-коррекцию",
            lambda: threading.Thread(target=calculate, daemon=True).start(),
        ).pack(anchor="e", pady=(12, 0))

    def _show_stage_placeholder(self, _: str | None = None) -> None:
        messagebox.showinfo(
            "Функция следующего этапа",
            "Отдельный расчёт для этой стадии пока не подключён. Данные не подменяются заглушкой.",
            parent=self,
        )


def smoke_snapshot(scenario_id: str) -> dict[str, Any]:
    """Headless smoke path used by CI and machines without a display."""
    scenario = load_scenario(PROJECT_ROOT / f"config/scenarios/{scenario_id}.json")
    result = run_model_demo(scenario_id)
    view = recommendation_to_view(result, scenario)
    return {
        "scenario_id": scenario_id,
        "status": view.status,
        "action_title": view.action_title,
        "baseline_upper": view.baseline_upper,
        "selected_upper": view.selected_upper,
        "constraints": [row.__dict__ for row in view.constraints],
        "run_id": result.run_id,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="НЕФТЕКОД desktop interface")
    parser.add_argument("--scenario", choices=tuple(SCENARIO_LABELS.values()), default="blend_risk")
    parser.add_argument(
        "--page",
        choices=("overview", "recommendation", "journal"),
        default="overview",
        help="initial screen",
    )
    parser.add_argument("--smoke", action="store_true", help="run calculation without opening Tk")
    args = parser.parse_args(argv)
    if args.smoke:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(smoke_snapshot(args.scenario), ensure_ascii=False, indent=2))
        return 0
    app = PetrolCodeApp(args.scenario, args.page)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConstraintRow",
    "DashboardView",
    "PetrolCodeApp",
    "format_action_shadow_payload",
    "format_v2_forecast_payload",
    "journal_entries",
    "main",
    "recommendation_to_view",
    "smoke_snapshot",
]

"""Tkinter desktop interface for model-demo, forecasts and gated action artifacts.

Historical effects and schema-1.2 forecasts are shadow-only. History replay can
show an actionable recommendation only when a verified action artifact is supplied.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Sequence

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
    history_interval_command,
    prepare_command,
    replay_v2_shadow_command,
    run_history_command,
    run_model_demo,
    validate_stage0,
)
from source.ui_charts import HistoryChart
from source.ui_data import (
    UiContext,
    UiHistoryReplayView,
    UiHybridBlendView,
    UiStageSnapshot,
    discover_ui_context,
    history_replay_to_view,
    ui_history_snapshot,
    ui_hybrid_snapshot,
    ui_stage_snapshot,
)
from source.ui_what_if import open_editor

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
RELEASE_DEMO_AS_OF = "2025-06-01T12:00:00+03:00"
STATUS_RU = {
    "empty": "ожидает расчёта",
    "ready": "готово",
    "error": "ошибка расчёта",
    "abstain": "отказ от управляющего действия",
    "hold": "сохранить режим",
    "recommend": "рекомендация",
    "pass": "в пределах",
    "fail": "превышение",
    "unknown": "неизвестно",
    "not_actionable": "не рекомендованы",
    "actionable": "разрешены моделью",
    "unavailable": "недоступны",
}

SCENARIO_LABELS = {
    "Риск и стоимость": "blend_tradeoff",
    "Нормальный режим": "blend_normal",
    "Повышенная сера": "blend_risk",
    "Риск T95": "blend_t95_risk",
    "Низкое цетановое число": "blend_cetane_risk",
    "Недостающие данные": "blend_missing",
}

PAGE_ALIASES = {"recommendation": "blend"}
PAGE_CHOICES = (
    "overview",
    "avt",
    "hydrotreating",
    "blend",
    "history",
    "journal",
    "recommendation",
)


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
    total_mass_t: float | None
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
        "kg/m3": "кг/м³",
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
        action_detail = (
            "Текущая рецептура проходит проверки; вариант улучшает модельные критерии."
            if result.baseline is not None and result.baseline.feasible
            else "Текущая рецептура нарушает одно или несколько ограничений качества."
        )
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
        total_mass_t=scenario.total_mass_t,
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
    control_unit = str(payload.get("control_unit", ""))
    delta = payload.get("proposed_delta")
    point = payload.get("predicted_sulfur", {})
    upper = payload.get("sulfur_upper", {})
    effect = payload.get("sulfur_change", {})
    rows = [
        "Модельный эффект по историческим эпизодам",
        "Не является советом по изменению уставки.",
        "",
        f"Текущая сера ПАК: {_format_value(baseline if isinstance(baseline, float) else None)}",
        f"Сценарий: {control} на {delta} {control_unit}".rstrip(),
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


def format_history_replay_view(view: UiHistoryReplayView) -> str:
    """Render a history replay result without hiding the raw journal payload."""
    rows = [
        "Исторический прогноз серы",
        "Прогноз рассчитан. Реальные уставки не изменяются."
        if view.status == "ready"
        else view.message,
        "",
        f"Статус: {STATUS_RU.get(view.recommendation_status or view.status, view.status)}",
        f"Модель: {view.model_id or '—'}",
        f"Момент данных: {view.as_of or '—'}",
        f"Сера, точечный прогноз: {_format_value(view.sulfur_point)}",
        f"Сера, верхняя граница: {_format_value(view.sulfur_upper)}",
        f"Проверка верхней границы: {STATUS_RU.get(view.upper_status, view.upper_status)}",
        f"Действия: {STATUS_RU.get(view.action_state, view.action_state)}",
        f"Выбранный тип: {view.selected_kind or '—'}",
        f"Коды причин: {', '.join(view.reason_codes) if view.reason_codes else '—'}",
    ]
    if view.issues:
        rows.append("Проблемы: " + " | ".join(view.issues))
    if view.journal_path:
        rows.append(f"Журнал: {view.journal_path}")
    if view.raw is not None:
        rows.extend(
            ("", "Raw recommendation JSON:", json.dumps(view.raw, ensure_ascii=False, indent=2))
        )
    return "\n".join(rows)


def format_hybrid_blend_view(view: UiHybridBlendView) -> str:
    """Render the hybrid sulfur-only blend panel in a compact form."""
    rows = [
        "Условное смешение по прогнозу серы (Hybrid sulfur-only blend)",
        "Рассчитана только сера. Остальные свойства смеси здесь не подтверждаются."
        if view.status == "ready"
        else view.message,
        "",
        f"Статус расчёта: {STATUS_RU.get(view.status, view.status)}",
        f"Сценарий: {view.scenario_id}",
        f"Модель: {view.model_id or '—'}",
        f"Состояние: {view.source_state_id or '—'}",
        f"Компонент A, сера point: {_format_value(view.component_sulfur_point)}",
        f"Компонент A, сера upper: {_format_value(view.component_sulfur_upper)}",
        f"Смесь, сера point: {_format_value(view.blend_sulfur_point)}",
        f"Смесь, сера upper: {_format_value(view.blend_sulfur_upper)}",
        "Проверка ограничения серы: "
        + STATUS_RU.get(view.constraint_status, view.constraint_status),
    ]
    if view.reason_codes:
        rows.append("Коды причин: " + ", ".join(view.reason_codes))
    if view.assumptions:
        rows.append("Допущения: " + " | ".join(view.assumptions))
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
        width = min(1536, max(1000, self.winfo_screenwidth() - 80))
        height = min(960, max(680, self.winfo_screenheight() - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(1000, 680)
        self.configure(bg=BG)
        self.option_add("*Font", "{Segoe UI} 11")
        self.scenario_var = tk.StringVar(value=self._scenario_label(initial_scenario))
        self.horizon_var = tk.StringVar(value="60 мин")
        self.status_var = tk.StringVar(value="Готово к расчёту")
        self._ui_events: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._closing = False
        self._interval_cancel = threading.Event()
        self._shared_dataset = tk.StringVar(value=discover_ui_context().latest_dataset or "")
        self._shared_as_of = tk.StringVar(value=RELEASE_DEMO_AS_OF)
        self._result: Recommendation | None = None
        self._scenario: ScenarioConfig | None = None
        self._calculation_request_id = 0
        self._history_request_id = 0
        self._hybrid_request_id = 0
        self._stage_request_id = 0
        self._history_view: UiHistoryReplayView | None = None
        self._history_export_payload: dict[str, Any] | None = None
        self._history_interval_text: str | None = None
        self._hybrid_view: UiHybridBlendView | None = None
        self._stage_snapshot_cache: dict[tuple[str, str, str], UiStageSnapshot] = {}
        self._overview_context: UiContext | None = None
        self._overview_key: tuple[str, str] | None = None
        self._overview_snapshots: dict[str, UiStageSnapshot] | None = None
        self._page = initial_page
        self._nav_buttons: dict[str, tk.Button] = {}
        self._build_styles()
        self._build_shell()
        self.calculate()
        self._drain_ui_events()

    def _post_ui(self, callback: Callable[[], None]) -> None:
        """Workers enqueue results without calling the Tcl interpreter."""
        if not self._closing:
            self._ui_events.put(callback)

    def _drain_ui_events(self) -> None:
        if self._closing:
            return
        while not self._ui_events.empty():
            self._ui_events.get_nowait()()
        self._poll_id = self.after(40, self._drain_ui_events)

    def destroy(self) -> None:
        self._closing = True
        self._interval_cancel.set()
        if hasattr(self, "_poll_id"):
            self.after_cancel(self._poll_id)
        super().destroy()

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
            ("avt", "АВТ"),
            ("hydrotreating", "Гидроочистка"),
            ("blend", "Смесь"),
            ("history", "История/ML"),
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
        page = PAGE_ALIASES.get(page, page)
        if self._page == "history":
            self._history_request_id += 1
            self._interval_cancel.set()
        if self._page == "blend" and page != "blend":
            self._hybrid_request_id += 1
        self._page = page
        self._set_nav(page)
        self._clear_body()
        if page == "overview":
            self._render_overview()
        elif page == "avt":
            self._render_stage_page("avt")
        elif page == "hydrotreating":
            self._render_stage_page("hydrotreating")
        elif page == "blend":
            self._render_recommendation()
        elif page == "history":
            self._render_history()
        elif page == "journal":
            self._render_journal()
        else:
            self._render_overview()

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

    @staticmethod
    def _field(
        parent: tk.Misc, label: str, variable: tk.StringVar, width: int | None = None
    ) -> None:
        tk.Label(parent, text=label, bg=BG, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        entry = tk.Entry(
            parent, textvariable=variable, bg=SURFACE, fg=TEXT, bd=1, width=width or 20
        )
        entry.pack(fill="x", pady=(4, 10), ipady=6)

    @staticmethod
    def _stage_tone(status: str) -> str:
        return {"fresh": "ok", "stale": "unknown", "missing": "bad"}.get(status, "unknown")

    @staticmethod
    def _widget_exists(widget: tk.Misc) -> bool:
        try:
            return bool(widget.winfo_exists())
        except tk.TclError:
            return False

    def _schedule_dialog_render(self, dialog: tk.Toplevel, render: Any, value: str) -> None:
        """Deliver a worker result only while its dialog still exists."""

        def finish() -> None:
            if self._widget_exists(dialog):
                try:
                    render(value)
                except tk.TclError:
                    return

        try:
            self._post_ui(finish)
        except tk.TclError:
            return

    def _render_stage_cards(self, parent: tk.Misc) -> None:
        if self._overview_key != (self._shared_dataset.get(), self._shared_as_of.get()):
            self._overview_context = None
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(16, 12))
        context = self._overview_context
        if context is None:
            for label in ("АВТ", "Гидроочистка", "История/ML"):
                card = self._surface(row)
                card.pack(side="left", fill="x", expand=True, padx=(0, 12), ipady=8)
                tk.Label(card, text=label, bg=SURFACE, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(
                    anchor="w", padx=18, pady=(14, 4)
                )
                tk.Label(
                    card,
                    text="Загрузка состояния…",
                    bg=SURFACE,
                    fg=MUTED,
                    wraplength=360,
                    justify="left",
                ).pack(anchor="w", padx=18, pady=(6, 14))
            self._load_overview_stage_cards(parent, row)
            return
        snapshots = self._overview_snapshots or {
            page: ui_stage_snapshot(page, context.latest_dataset)
            for page in ("avt", "hydrotreating")
        }
        for page, label in (("avt", "АВТ"), ("hydrotreating", "Гидроочистка")):
            snapshot = snapshots[page]
            card = self._surface(row)
            card.pack(side="left", fill="x", expand=True, padx=(0, 12), ipady=8)
            tk.Label(card, text=label, bg=SURFACE, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(
                anchor="w", padx=18, pady=(14, 4)
            )
            tk.Label(
                card,
                text=(
                    f"свежие {snapshot.fresh_count} · устаревшие {snapshot.stale_count} · "
                    f"нет данных {snapshot.missing_count}"
                ),
                bg=SURFACE,
                fg=TEXT,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w", padx=18)
            tk.Label(
                card,
                text=snapshot.message,
                bg=SURFACE,
                fg=MUTED,
                wraplength=360,
                justify="left",
            ).pack(anchor="w", padx=18, pady=(6, 14))
            self._secondary_button(card, "Открыть", partial(self.show_page, page)).pack(
                anchor="w", padx=18, pady=(0, 14)
            )
        card = self._surface(row)
        card.pack(side="left", fill="x", expand=True, ipady=8)
        tk.Label(card, text="История/ML", bg=SURFACE, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=18, pady=(14, 4)
        )
        forecast_count = len(context.forecast_artifacts)
        action_count = len(context.action_artifacts)
        dataset_text = "dataset найден" if context.latest_dataset else "dataset отсутствует"
        tk.Label(
            card,
            text=f"{dataset_text} · forecast {forecast_count} · action {action_count}",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=18)
        tk.Label(
            card,
            text=(
                "Forecast работает read-only; actionable совет включается только "
                "verified action artifact."
            ),
            bg=SURFACE,
            fg=MUTED,
            wraplength=360,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(6, 14))
        self._secondary_button(card, "Открыть", partial(self.show_page, "history")).pack(
            anchor="w", padx=18, pady=(0, 14)
        )

    def _load_overview_stage_cards(self, parent: tk.Misc, row: tk.Misc) -> None:
        self._stage_request_id += 1
        request_id = self._stage_request_id
        dataset = self._shared_dataset.get()
        as_of = self._shared_as_of.get()

        def work() -> None:
            context = discover_ui_context()
            snapshots = {
                page: ui_stage_snapshot(page, dataset, as_of) for page in ("avt", "hydrotreating")
            }

            def finish() -> None:
                if request_id != self._stage_request_id or self._page != "overview":
                    return
                if not self._widget_exists(parent):
                    return
                self._overview_context = context
                self._overview_key = (dataset, as_of)
                self._overview_snapshots = snapshots
                for key, snapshot in snapshots.items():
                    self._stage_snapshot_cache[
                        (key, context.latest_dataset or "", snapshot.as_of or "")
                    ] = snapshot
                row.destroy()
                self._render_stage_cards(parent)

            try:
                self._post_ui(finish)
            except tk.TclError:
                return

        threading.Thread(target=work, daemon=True).start()

    def _render_stage_page(self, page_key: str) -> None:
        page = self._page_container()
        title = "АВТ" if page_key == "avt" else "Гидроочистка"
        subtitle = (
            "Read-only контекст установки АВТ: значения, единицы, источник и свежесть."
            if page_key == "avt"
            else "Read-only контекст гидроочистки: качество, P8/F19 readiness и газовый контур."
        )
        tk.Label(page, text=title, bg=BG, fg=TEXT, font=("Segoe UI", 28, "bold")).pack(anchor="w")
        tk.Label(page, text=subtitle, bg=BG, fg=MUTED, font=("Segoe UI", 11)).pack(
            anchor="w", pady=(2, 18)
        )

        dataset_var = self._shared_dataset
        as_of_var = self._shared_as_of
        controls = self._surface(page)
        controls.pack(fill="x", pady=(0, 14), padx=0)
        inner = tk.Frame(controls, bg=SURFACE)
        inner.pack(fill="x", padx=20, pady=16)
        left = tk.Frame(inner, bg=SURFACE)
        left.pack(side="left", fill="x", expand=True, padx=(0, 14))
        right = tk.Frame(inner, bg=SURFACE)
        right.pack(side="left", fill="x", expand=True)
        self._field(left, "Исторические данные", dataset_var)
        self._field(right, "Момент данных (ISO с timezone)", as_of_var)
        content = tk.Frame(page, bg=BG)
        content.pack(fill="both", expand=True)

        def draw(snapshot: UiStageSnapshot | None = None) -> None:
            for child in content.winfo_children():
                child.destroy()
            if snapshot is None:
                self._text_content(content, "Загрузка исторических данных…")
                self._stage_request_id += 1
                request_id = self._stage_request_id
                dataset = dataset_var.get() or None
                as_of = as_of_var.get().strip() or None

                def work() -> None:
                    result = ui_stage_snapshot(page_key, dataset, as_of)

                    def finish() -> None:
                        if request_id != self._stage_request_id or self._page != page_key:
                            return
                        if not self._widget_exists(content):
                            return
                        if result.as_of:
                            as_of_var.set(result.as_of)
                        draw(result)

                    try:
                        self._post_ui(finish)
                    except tk.TclError:
                        return

                threading.Thread(target=work, daemon=True).start()
                return
            self._render_stage_snapshot(content, snapshot)

        self._primary_button(inner, "Обновить", draw).pack(side="right", padx=(14, 0), pady=20)
        draw()

    def _render_stage_snapshot(self, parent: tk.Misc, snapshot: UiStageSnapshot) -> None:
        summary = self._surface(parent)
        summary.pack(fill="x", pady=(0, 14))
        for label, value in (
            ("Набор данных", snapshot.dataset_id or "—"),
            ("Момент данных", snapshot.as_of or "—"),
            ("Свежие", str(snapshot.fresh_count)),
            ("Устаревшие", str(snapshot.stale_count)),
            ("Нет данных", str(snapshot.missing_count)),
        ):
            cell = tk.Frame(summary, bg=SURFACE)
            cell.pack(side="left", fill="x", expand=True, padx=18, pady=16)
            tk.Label(cell, text=label, bg=SURFACE, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(
                anchor="w"
            )
            tk.Label(cell, text=value, bg=SURFACE, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(
                anchor="w", pady=(3, 0)
            )
        if snapshot.status != "ready":
            self._text_content(parent, snapshot.message)
            return
        table = ttk.Treeview(
            parent,
            columns=("group", "signal", "label", "value", "source", "age", "status", "reason"),
            show="headings",
            height=max(8, min(18, len(snapshot.rows))),
            style="Petrol.Treeview",
        )
        for key, title, width in (
            ("group", "Группа", 150),
            ("signal", "Signal", 120),
            ("label", "Смысл", 310),
            ("value", "Значение", 120),
            ("source", "Источник", 95),
            ("age", "Возраст", 95),
            ("status", "Статус", 95),
            ("reason", "Ограничение", 280),
        ):
            table.heading(key, text=title)
            table.column(key, width=width, anchor="w", stretch=True)
        table.tag_configure("ok", foreground=GREEN)
        table.tag_configure("bad", foreground=RED)
        table.tag_configure("unknown", foreground=AMBER)
        for row in snapshot.rows:
            age = "—" if row.age_minutes is None else f"{row.age_minutes:.0f} мин"
            issue = row.issue or row.read_only_reason
            table.insert(
                "",
                "end",
                values=(
                    row.group,
                    row.signal_id,
                    row.label,
                    row.value_text,
                    row.source,
                    age,
                    row.freshness,
                    issue,
                ),
                tags=(self._stage_tone(row.freshness),),
            )
        table.pack(fill="both", expand=True)

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
        self._secondary_button(page, "Синтетический what-if", self.open_what_if_dialog).pack(
            anchor="w", pady=(10, 0)
        )

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
        self._render_stage_cards(page)

        stages = tk.Frame(page, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        stages.pack(fill="x", pady=(16, 12), ipady=4)
        for text, target in (
            ("АВТ", "avt"),
            ("Гидроочистка", "hydrotreating"),
            ("Смесь", "blend"),
        ):
            command = partial(self.show_page, target)
            button = tk.Button(
                stages,
                text=text,
                command=command,
                bg=TEAL,
                fg="white",
                activebackground=TEAL_HOVER,
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
        self._info_row(parent, "Сценарий", self._scenario.id if self._scenario else "—")
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

    def _render_history(self) -> None:
        page = self._page_container()
        tk.Label(page, text="История и ML", bg=BG, fg=TEXT, font=("Segoe UI", 28, "bold")).pack(
            anchor="w"
        )
        tk.Label(
            page,
            text=(
                "Исторические данные и прогноз серы в режиме только чтение. "
                "Реальные действия не рекомендуются."
            ),
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(2, 18))
        context = discover_ui_context()
        dataset_var = self._shared_dataset
        model_var = tk.StringVar(
            value=context.forecast_artifacts[0].path if context.forecast_artifacts else ""
        )
        action_var = tk.StringVar(
            value=context.action_artifacts[0].path if context.action_artifacts else ""
        )
        as_of_var = self._shared_as_of
        interval_to_var = tk.StringVar(value="2025-06-01T18:00:00+03:00")
        fields = self._surface(page)
        fields.pack(fill="x", pady=(0, 14))
        grid = tk.Frame(fields, bg=SURFACE)
        grid.pack(fill="x", padx=20, pady=16)
        for column in range(2):
            grid.grid_columnconfigure(column, weight=1)
        left = tk.Frame(grid, bg=SURFACE)
        left.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        right = tk.Frame(grid, bg=SURFACE)
        right.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        self._field(left, "Момент данных (ISO с timezone)", as_of_var)
        self._field(right, "Конец исторического интервала (ISO с timezone)", interval_to_var)

        def settings(parent: tk.Misc) -> None:
            self._field(parent, "Исторические данные", dataset_var)
            self._field(parent, "Прогноз серы", model_var)
            self._field(parent, "Артефакт действий (не используется)", action_var)

        self._accordion(page, "Дополнительные настройки данных и модели", settings).pack(
            fill="x", pady=(0, 10)
        )
        summary = tk.StringVar(
            value="Укажите время и запустите расчёт. Реальные действия отключены."
        )
        tk.Label(
            page, textvariable=summary, bg=BG, fg=TEXT, anchor="w", justify="left", wraplength=1100
        ).pack(fill="x", pady=8)
        chart = HistoryChart(page)
        chart.pack(fill="x", pady=(0, 10))
        output = tk.Text(page, height=7, bg=SURFACE, fg=TEXT, bd=1, wrap="word")
        output.pack(fill="both", expand=True)

        def render(view: UiHistoryReplayView) -> None:
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("1.0", format_history_replay_view(view))
            output.configure(state="disabled")
            chart.set_payload(view.raw)
            summary.set(
                f"Прогноз: {_format_value(view.sulfur_point)} · "
                f"Верхняя оценка: {_format_value(view.sulfur_upper)}. "
                "Реальные действия отключены."
                if view.status == "ready"
                else view.message
            )

        def calculate(
            request_id: int,
            dataset: str | None,
            model: str | None,
            as_of: str,
            action_model: str | None,
        ) -> None:
            view = ui_history_snapshot(dataset, model, as_of, action_model)

            def finish() -> None:
                if (
                    request_id != self._history_request_id
                    or self._page != "history"
                    or not self._widget_exists(output)
                ):
                    return
                self._history_view = view
                self._history_export_payload = view.raw
                self._history_interval_text = None
                if output.winfo_exists():
                    render(view)

            self._post_ui(finish)

        def start_calculation() -> None:
            self._history_request_id += 1
            self._interval_cancel.set()
            self._history_export_payload = None
            self._history_view = None
            self._history_interval_text = None
            chart.set_payload(None)
            summary.set("Расчёт исторического прогноза…")
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("1.0", "Расчёт исторического прогноза…")
            output.configure(state="disabled")
            threading.Thread(
                target=calculate,
                args=(
                    self._history_request_id,
                    dataset_var.get().strip() or None,
                    model_var.get().strip() or None,
                    as_of_var.get(),
                    action_var.get().strip() or None,
                ),
                daemon=True,
            ).start()

        def calculate_interval() -> None:
            self._history_request_id += 1
            self._interval_cancel.set()
            cancel = threading.Event()
            self._interval_cancel = cancel
            self._history_export_payload = None
            self._history_view = None
            self._history_interval_text = None
            chart.set_payload(None)
            request_id = self._history_request_id
            dataset = dataset_var.get().strip()
            model = model_var.get().strip()
            try:
                start = datetime.fromisoformat(as_of_var.get().replace("Z", "+00:00"))
                end = datetime.fromisoformat(interval_to_var.get().replace("Z", "+00:00"))
                if start.tzinfo is None or end.tzinfo is None:
                    raise ValueError("обе границы интервала должны содержать timezone")
            except (TypeError, ValueError) as exc:
                self._history_interval_text = f"Ошибка ввода интервала: {exc}"
                summary.set(self._history_interval_text)
                output.configure(state="normal")
                output.delete("1.0", "end")
                output.insert("1.0", self._history_interval_text)
                output.configure(state="disabled")
                return

            def progress(done: int, total: int) -> None:
                def show_progress() -> None:
                    if request_id != self._history_request_id or not self._widget_exists(output):
                        return
                    output.configure(state="normal")
                    summary.set(
                        f"Исторический интервал: {done} / {total} точек. "
                        "Отмена — после текущей точки."
                    )
                    output.delete("1.0", "end")
                    output.insert(
                        "1.0",
                        f"Исторический интервал: {done} / {total} точек.\n"
                        "Отмена завершится после текущей точки.\n"
                        "Начальная загрузка данных может занять несколько секунд.",
                    )
                    output.configure(state="disabled")

                self._post_ui(show_progress)

            progress(0, 0)

            def work() -> None:
                payload = None
                try:
                    payload = history_interval_command(
                        dataset, model, start, end, progress=progress, cancelled=cancel.is_set
                    )
                    title = (
                        "Интервал отменён. Частичный результат"
                        if payload.get("cancelled")
                        else "Исторический интервал"
                    )
                    text = title + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
                except Exception as exc:
                    text = f"Ошибка интервала: {exc}"

                def finish() -> None:
                    if request_id != self._history_request_id or self._page != "history":
                        return
                    if self._widget_exists(output):
                        self._history_export_payload = payload
                        self._history_interval_text = text
                        chart.set_payload(payload)
                        summary.set(
                            (
                                "Частичный результат отменённого интервала"
                                if payload.get("cancelled")
                                else "Интервал рассчитан"
                            )
                            + f": {payload.get('points', 0)} точек. Реальные действия отключены."
                            if payload is not None
                            else text
                        )
                        output.configure(state="normal")
                        output.delete("1.0", "end")
                        output.insert("1.0", text)
                        output.configure(state="disabled")

                try:
                    self._post_ui(finish)
                except tk.TclError:
                    return

            threading.Thread(target=work, daemon=True).start()

        actions = tk.Frame(page, bg=BG)
        actions.pack(fill="x", pady=(0, 12), before=chart)
        self._primary_button(
            actions,
            "Рассчитать на выбранный момент",
            start_calculation,
        ).pack(side="left")
        self._secondary_button(actions, "Журнал", lambda: self.show_page("journal")).pack(
            side="left", padx=12
        )
        self._secondary_button(actions, "Рассчитать интервал", calculate_interval).pack(side="left")
        self._secondary_button(
            actions, "Отменить интервал", lambda: self._interval_cancel.set()
        ).pack(side="left")
        self._secondary_button(
            actions, "Экспортировать этот результат", self.export_history_result
        ).pack(side="left", padx=12)
        buttons = [child for child in actions.winfo_children() if isinstance(child, tk.Button)]
        for button in buttons:
            button.pack_forget()
        for index, button in enumerate(buttons):
            button.grid(row=index // 3, column=index % 3, sticky="ew", padx=4, pady=4)
        for column in range(3):
            actions.grid_columnconfigure(column, weight=1)
        render(
            self._history_view
            or UiHistoryReplayView(
                status="empty",
                message="Данные и модель уже выбраны. Укажите время и запустите расчёт.",
                recommendation_status=None,
                scenario_id=None,
                model_id=None,
                as_of=None,
                sulfur_point=None,
                sulfur_upper=None,
                upper_status="unknown",
                selected_kind=None,
                action_state="unavailable",
                reason_codes=(),
                issues=(),
                journal_path=None,
                raw=None,
            )
        )
        if self._history_interval_text is not None:
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("1.0", self._history_interval_text)
            output.configure(state="disabled")
            chart.set_payload(self._history_export_payload)
            summary.set(
                "Частичный интервал"
                if self._history_export_payload and self._history_export_payload.get("cancelled")
                else "Результат последнего интервала"
            )

    def _render_hybrid_panel(self, parent: tk.Misc) -> None:
        panel = self._surface(parent)
        panel.pack(fill="x", pady=(16, 0))
        tk.Label(
            panel,
            text="Hybrid: АВТ → гидроочистка → смесь",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", padx=26, pady=(16, 4))
        tk.Label(
            panel,
            text=(
                "Компонент A берётся только из history forecast. Если forecast недоступен, "
                "UI не подставляет synthetic production-значение."
            ),
            bg=SURFACE,
            fg=MUTED,
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", padx=26, pady=(0, 12))
        context = discover_ui_context()
        dataset_var = self._shared_dataset
        model_var = tk.StringVar(
            value=context.forecast_artifacts[0].path if context.forecast_artifacts else ""
        )
        as_of_var = self._shared_as_of
        form = tk.Frame(panel, bg=SURFACE)
        form.pack(fill="x", padx=26)
        left = tk.Frame(form, bg=SURFACE)
        left.pack(side="left", fill="x", expand=True, padx=(0, 10))
        middle = tk.Frame(form, bg=SURFACE)
        middle.pack(side="left", fill="x", expand=True, padx=10)
        right = tk.Frame(form, bg=SURFACE)
        right.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self._field(left, "Исторические данные", dataset_var)
        self._field(middle, "Прогноз серы", model_var)
        self._field(right, "Момент данных (ISO с timezone)", as_of_var)
        output = tk.Text(panel, height=10, bg="#FAFBFB", fg=TEXT, bd=0, wrap="word")
        output.pack(fill="x", padx=26, pady=(4, 16))

        def render(view: UiHybridBlendView) -> None:
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("1.0", format_hybrid_blend_view(view))
            output.configure(state="disabled")

        def calculate(request_id: int, dataset: str, model: str, as_of: str) -> None:
            view = ui_hybrid_snapshot(dataset or None, model or None, as_of)

            def finish() -> None:
                if request_id != self._hybrid_request_id or self._page != "blend":
                    return
                if self._widget_exists(output):
                    self._hybrid_view = view
                    render(view)

            try:
                self._post_ui(finish)
            except tk.TclError:
                return

        def start_calculation() -> None:
            self._hybrid_request_id += 1
            request_id = self._hybrid_request_id
            threading.Thread(
                target=calculate,
                args=(
                    request_id,
                    dataset_var.get().strip(),
                    model_var.get().strip(),
                    as_of_var.get(),
                ),
                daemon=True,
            ).start()

        self._secondary_button(
            panel,
            "Рассчитать hybrid",
            start_calculation,
        ).pack(anchor="e", padx=26, pady=(0, 16))
        self._secondary_button(panel, "Экспортировать hybrid", self.export_hybrid_result).pack(
            anchor="e", padx=26, pady=(0, 16)
        )
        render(
            self._hybrid_view
            or UiHybridBlendView(
                status="empty",
                message="Hybrid не рассчитан.",
                scenario_id="hybrid_blend",
                source_state_id=None,
                model_id=None,
                component_sulfur_point=None,
                component_sulfur_upper=None,
                blend_sulfur_point=None,
                blend_sulfur_upper=None,
                constraint_status="unknown",
                assumptions=(),
                reason_codes=(),
            )
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
            text="⚠  Синтетический what-if"
            if self._scenario and self._scenario.id.endswith("_what_if")
            else "⚠  Модельный режим",
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
        self._render_hybrid_panel(page)
        footer = tk.Frame(page, bg=BG)
        footer.pack(fill="x", pady=(18, 0))
        self._primary_button(footer, "Скачать расчёт", self.export_result).pack(side="left")
        self._secondary_button(footer, "Синтетический what-if", self.open_what_if_dialog).pack(
            side="left", padx=10
        )
        self._secondary_button(footer, "Открыть журнал", lambda: self.show_page("journal")).pack(
            side="left", padx=14
        )
        tk.Label(
            footer,
            text="НЕФТЕКОД  v1.1  |  prototype",
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

            def fraction_text(fraction: float) -> str:
                percent = f"{fraction * 100:.4f}".rstrip("0").rstrip(".").replace(".", ",")
                if view.total_mass_t is None:
                    return f"{percent} %"
                mass = (
                    f"{fraction * view.total_mass_t:.4f}".rstrip("0").rstrip(".").replace(".", ",")
                )
                return f"{percent} % · {mass} т"

            for component in sorted(view.current_fractions):
                table.insert(
                    "",
                    "end",
                    values=(
                        f"Компонент {component}",
                        fraction_text(view.current_fractions[component]),
                        (
                            "Рецептура не выбрана"
                            if view.status == RecommendationStatus.ABSTAIN.value
                            else fraction_text(view.proposed_fractions.get(component, 0.0))
                        ),
                    ),
                )
            table.insert(
                "",
                "end",
                values=(
                    "Цетаноповышающая присадка",
                    fraction_text(view.current_additive_fraction),
                    (
                        "Рецептура не выбрана"
                        if view.status == RecommendationStatus.ABSTAIN.value
                        else fraction_text(view.proposed_additive_fraction)
                    ),
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
        labels: list[str] = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                as_of = str(payload.get("as_of", "—"))[:19]
                scenario = str(payload.get("scenario_id", "расчёт"))
                status = str(payload.get("status", "—"))
                run_id = str(payload.get("run_id", path.parent.name))[:8]
                label = f"{as_of} · {scenario} · {status} · {run_id}"
            except (OSError, json.JSONDecodeError):
                label = f"{path.parent.name} · повреждённая запись"
            labels.append(label)
            listbox.insert("end", label)

        def select(_: tk.Event[Any] | None = None) -> None:
            selection = listbox.curselection()
            if not selection:
                return
            path = paths[selection[0]]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                metadata_path = path.parent / "metadata.json"
                metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.is_file()
                    else {}
                )
                summary = (
                    f"Запуск: {payload.get('run_id', path.parent.name)}\n"
                    f"Сценарий: {payload.get('scenario_id', '—')}\n"
                    f"Момент данных: {payload.get('as_of', '—')}\n"
                    f"Статус: {payload.get('status', '—')}\n"
                    f"Модель: {payload.get('model_id', '—')}\n"
                    f"Папка доказательств: {path.parent}\n\n"
                    "Краткий результат:\n"
                    f"{payload.get('explanation', '—')}\n\n"
                    "Полный result.json:\n"
                )
                rendered = summary + json.dumps(payload, ensure_ascii=False, indent=2)
                if metadata:
                    rendered += "\n\nmetadata.json:\n" + json.dumps(
                        metadata, ensure_ascii=False, indent=2
                    )
            except (OSError, json.JSONDecodeError) as exc:
                rendered = f"Не удалось открыть текущий запуск: {exc}"
            detail.configure(state="normal")
            detail.delete("1.0", "end")
            detail.insert("1.0", rendered)
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
    def _text_content(parent: tk.Misc, text: str) -> None:
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
        self._secondary_button(
            row, "Исторический эффект P8/F19 (не совет)", self.open_action_shadow_dialog
        ).pack(side="left", padx=(10, 0))

    def _view(self) -> DashboardView | None:
        if self._result is None or self._scenario is None:
            return None
        return recommendation_to_view(self._result, self._scenario)

    def open_what_if_dialog(self) -> None:
        preset = load_scenario(
            PROJECT_ROOT / f"config/scenarios/{SCENARIO_LABELS[self.scenario_var.get()]}.json"
        )
        current = (
            self._scenario
            if self._scenario and self._scenario.id == preset.id + "_what_if"
            else preset
        )

        def recalculate(scenario: ScenarioConfig) -> None:
            self.calculate(scenario)

        open_editor(self, preset, current, recalculate)

    def calculate(self, scenario_override: ScenarioConfig | None = None) -> None:
        self._calculation_request_id += 1
        request_id = self._calculation_request_id
        scenario_id = SCENARIO_LABELS[self.scenario_var.get()]
        self.status_var.set("Выполняется расчёт…")
        self._result = None
        self._scenario = None
        self.show_page("blend" if scenario_override is not None else self._page)

        def work() -> None:
            try:
                scenario = scenario_override or load_scenario(
                    PROJECT_ROOT / f"config/scenarios/{scenario_id}.json"
                )
                result = run_model_demo(scenario_id, scenario_override=scenario_override)
            except Exception as exc:  # UI boundary: render backend failure without crashing Tk.
                self._post_ui(partial(self._calculation_failed, request_id, exc))
                return
            self._post_ui(lambda: self._calculation_done(request_id, result, scenario))

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

    def export_history_result(self) -> None:
        payload = self._history_export_payload
        if payload is None:
            messagebox.showinfo(
                "Нет исторического результата",
                "Сначала выполните исторический прогноз.",
                parent=self,
            )
            return
        destination = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить исторический прогноз",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"),),
            initialfile=f"neftekod-history-{payload.get('model_id') or 'result'}.json",
        )
        if not destination:
            return
        try:
            Path(destination).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
                newline="\n",
            )
        except OSError as exc:
            messagebox.showerror("Не удалось сохранить", str(exc), parent=self)

    def export_hybrid_result(self) -> None:
        if self._hybrid_view is None or self._hybrid_view.status == "empty":
            messagebox.showinfo(
                "Нет hybrid-результата",
                "Сначала выполните условное смешение по прогнозу серы.",
                parent=self,
            )
            return
        destination = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить hybrid-результат",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"),),
            initialfile="neftekod-hybrid-result.json",
        )
        if not destination:
            return
        try:
            Path(destination).write_text(
                json.dumps(asdict(self._hybrid_view), ensure_ascii=False, indent=2),
                encoding="utf-8",
                newline="\n",
            )
        except OSError as exc:
            messagebox.showerror("Не удалось сохранить", str(exc), parent=self)

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
        if not destination:
            return
        try:
            payload = self._result.model_dump(mode="json")
            if self._scenario is not None and self._scenario.id.endswith("_what_if"):
                payload["synthetic_scenario"] = self._scenario.model_dump(mode="json")
            Path(destination).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
            )
        except OSError as exc:
            messagebox.showerror("Не удалось сохранить", str(exc), parent=self)

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
                self._post_ui(partial(self._preparation_failed, exc))
                return
            self._post_ui(partial(self._preparation_done, result))

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
        tk.Label(fields, text="Исторические данные", bg=BG, fg=TEXT).pack(anchor="w")
        dataset_var = self._shared_dataset
        tk.Entry(fields, textvariable=dataset_var, bg=SURFACE, fg=TEXT, bd=1).pack(
            fill="x", pady=(4, 14), ipady=7
        )
        tk.Label(fields, text="Момент данных (ISO с timezone)", bg=BG, fg=TEXT).pack(anchor="w")
        as_of_var = self._shared_as_of
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
        dataset_var = self._shared_dataset
        # Action-shadow artifacts are not part of the live release selection.
        model_var = tk.StringVar(value="")
        control_var = tk.StringVar(value="ht:P8")
        delta_var = tk.StringVar(value="0.001")
        at_var = self._shared_as_of
        for label, variable in (
            ("Исторические данные", dataset_var),
            ("Исследовательский action-артефакт", model_var),
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
                "Расчёт показывает модельный эффект по наблюдавшимся эпизодам. P8 — "
                "перепад давления реактора Р-202 (МПа), F19 — расход бензина в К-201 (т/ч). "
                "Это не совет по изменению уставки и не команда оборудованию."
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

        def calculate(values: dict[str, str]) -> None:
            try:
                at = datetime.fromisoformat(values["at_var"].replace("Z", "+00:00"))
                if at.tzinfo is None:
                    raise ValueError("время должно содержать timezone")
                result = action_shadow_estimate_command(
                    values["dataset_var"],
                    values["model_var"],
                    values["control_var"],
                    float(values["delta_var"]),
                    at,
                )
                text = format_action_shadow_payload(result)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self._schedule_dialog_render(dialog, render, text)

        self._primary_button(
            fields,
            "Рассчитать исследовательский сценарий",
            lambda: threading.Thread(
                target=calculate,
                args=(
                    {
                        "at_var": at_var.get(),
                        "dataset_var": dataset_var.get(),
                        "model_var": model_var.get(),
                        "control_var": control_var.get(),
                        "delta_var": delta_var.get(),
                    },
                ),
                daemon=True,
            ).start(),
        ).pack(anchor="e", pady=(12, 0))

    def open_history_forecast_dialog(self) -> None:
        """Replay a trusted local sulfur artifact at one historical timestamp."""
        dialog = tk.Toplevel(self)
        dialog.title("Исторический прогноз серы")
        dialog.geometry("760x640")
        dialog.configure(bg=BG)
        dialog.transient(self)
        fields = tk.Frame(dialog, bg=BG)
        fields.pack(fill="both", expand=True, padx=26, pady=22)
        context = discover_ui_context()
        dataset_var = self._shared_dataset
        model_var = tk.StringVar(
            value=context.forecast_artifacts[0].path if context.forecast_artifacts else ""
        )
        action_model_var = tk.StringVar(
            value=context.action_artifacts[0].path if context.action_artifacts else ""
        )
        at_var = self._shared_as_of
        for label, variable in (
            ("Исторические данные", dataset_var),
            ("Прогноз серы", model_var),
            (
                "Проверенный action-артефакт (необязательно)",
                action_model_var,
            ),
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
                "Без verified action artifact это предупреждение, а не совет по уставкам."
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

        def calculate(values: dict[str, str]) -> None:
            try:
                at = datetime.fromisoformat(values["at_var"].replace("Z", "+00:00"))
                if at.tzinfo is None:
                    raise ValueError("время должно содержать timezone")
                action_model = values["action_model_var"].strip() or None
                view = ui_history_snapshot(
                    values["dataset_var"],
                    values["model_var"],
                    at,
                    action_model,
                )
                text = format_history_replay_view(view)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self._schedule_dialog_render(dialog, render, text)

        self._primary_button(
            fields,
            "Рассчитать прогноз",
            lambda: threading.Thread(
                target=calculate,
                args=(
                    {
                        "at_var": at_var.get(),
                        "action_model_var": action_model_var.get(),
                        "dataset_var": dataset_var.get(),
                        "model_var": model_var.get(),
                    },
                ),
                daemon=True,
            ).start(),
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
        context = discover_ui_context()
        dataset_var = self._shared_dataset
        model_var = tk.StringVar(value=context.v2_artifacts[0].path if context.v2_artifacts else "")
        at_var = self._shared_as_of
        for label, variable in (
            ("Исторические данные", dataset_var),
            ("Shadow-артефакт v2", model_var),
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

        def calculate(values: dict[str, str]) -> None:
            try:
                at = datetime.fromisoformat(values["at_var"].replace("Z", "+00:00"))
                if at.tzinfo is None:
                    raise ValueError("время должно содержать timezone")
                result = replay_v2_shadow_command(values["dataset_var"], values["model_var"], at)
                text = format_v2_forecast_payload(result)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self._schedule_dialog_render(dialog, render, text)

        self._primary_button(
            fields,
            "Рассчитать episode forecast",
            lambda: threading.Thread(
                target=calculate,
                args=(
                    {
                        "at_var": at_var.get(),
                        "dataset_var": dataset_var.get(),
                        "model_var": model_var.get(),
                    },
                ),
                daemon=True,
            ).start(),
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
        context = discover_ui_context()
        dataset_var = self._shared_dataset
        model_var = tk.StringVar(
            value=context.forecast_artifacts[0].path if context.forecast_artifacts else ""
        )
        for label, variable in (
            ("Исторические данные", dataset_var),
            ("Прогноз ПАК", model_var),
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

        def calculate(values: dict[str, str]) -> None:
            try:
                result = evaluate_lims_correction_command(
                    values["dataset_var"], values["model_var"]
                )
                text = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as exc:
                text = f"Ошибка: {exc}"
            self._schedule_dialog_render(dialog, render, text)

        self._primary_button(
            fields,
            "Оценить LIMS-коррекцию",
            lambda: threading.Thread(
                target=calculate,
                args=({"dataset_var": dataset_var.get(), "model_var": model_var.get()},),
                daemon=True,
            ).start(),
        ).pack(anchor="e", pady=(12, 0))


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


def history_smoke_snapshot(
    dataset: str | Path,
    model: str | Path,
    as_of: str,
    *,
    trusted_model: bool = True,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Headless history forecast path used by tests and demo machines."""
    timestamp = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    result = run_history_command(
        dataset,
        model,
        timestamp,
        trusted_model=trusted_model,
        run_dir=run_dir,
    )
    return {
        "status": result.status.value,
        "model_id": result.model_id,
        "reason_codes": list(result.reason_codes),
        "run_id": result.run_id,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="НЕФТЕКОД desktop interface")
    parser.add_argument("--scenario", choices=tuple(SCENARIO_LABELS.values()), default="blend_risk")
    parser.add_argument(
        "--page",
        choices=PAGE_CHOICES,
        default="overview",
        help="initial screen",
    )
    parser.add_argument("--smoke", action="store_true", help="run calculation without opening Tk")
    parser.add_argument("--history-smoke", action="store_true", help="run history without Tk")
    parser.add_argument("--dataset", default=None, help="prepared dataset for --history-smoke")
    parser.add_argument("--model", default=None, help="model artifact for --history-smoke")
    parser.add_argument("--as-of", default="2026-01-15T09:00:00Z", help="history as_of")
    parser.add_argument("--run-dir", default=None, help="history smoke journal directory")
    args = parser.parse_args(argv)
    if args.history_smoke:
        if args.dataset is None or args.model is None:
            parser.error("--history-smoke requires --dataset and --model")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(
            json.dumps(
                history_smoke_snapshot(args.dataset, args.model, args.as_of, run_dir=args.run_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
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
    "history_smoke_snapshot",
    "format_action_shadow_payload",
    "format_history_replay_view",
    "format_hybrid_blend_view",
    "format_v2_forecast_payload",
    "history_replay_to_view",
    "journal_entries",
    "main",
    "recommendation_to_view",
    "smoke_snapshot",
    "ui_history_snapshot",
    "ui_hybrid_snapshot",
    "ui_stage_snapshot",
]

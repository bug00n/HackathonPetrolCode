"""Synthetic-only editor; use the existing scenario contracts and decision engine."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from source.contracts import ScenarioConfig

COMPONENT_FIELDS = (
    ("sulfur.value", "Сера, мг/кг"),
    ("sulfur.upper", "Сера сверху, мг/кг"),
    ("t95.value", "T95, °C"),
    ("t95.upper", "T95 сверху, °C"),
    ("cetane_number.value", "Цетановое число"),
    ("cetane_number.lower", "Цетановое снизу"),
    ("available_mass_t", "Запас, т"),
    ("cost_proxy_per_t", "Стоимость, индекс/т"),
    ("risk_index", "Риск, от 0 до 1"),
    ("fraction", "Текущая доля, %"),
)
ADDITIVE_FIELDS = (
    ("total_mass_t", "Масса партии, т"),
    ("dose", "Текущая присадка, %"),
    ("max_dose", "Максимум присадки, % (1, 2 или 3)"),
    ("additive_stock", "Запас присадки, т"),
    ("additive_cost", "Стоимость присадки, индекс/т"),
    ("gain_1", "Прирост цетанового числа при 1%"),
    ("gain_2", "Прирост цетанового числа при 2%"),
    ("gain_3", "Прирост цетанового числа при 3%"),
)


def editor_values(scenario: ScenarioConfig) -> dict[str, str]:
    """Represent null quality explicitly as an empty field, never as zero."""
    values: dict[str, str] = {}
    for component in scenario.blend_components:
        for field, _ in COMPONENT_FIELDS:
            value: float | None
            if field == "fraction":
                value = scenario.current_blend_mass_fractions[component.id] * 100
            elif "." in field:
                metric, bound = field.split(".")
                value = getattr(getattr(component, metric), bound, None)
            else:
                value = getattr(component, field)
            values[f"{component.id}.{field}"] = "" if value is None else f"{value:.12g}"
    additive = scenario.cetane_additive
    if additive is None:
        raise ValueError("В сценарии нет модели присадки")
    values.update(
        total_mass_t=f"{scenario.total_mass_t:.12g}",
        dose=f"{scenario.current_additive_mass_fraction * 100:.12g}",
        max_dose=f"{additive.max_mass_fraction * 100:.12g}",
        additive_stock=f"{additive.available_mass_t:.12g}",
        additive_cost=f"{additive.cost_proxy_per_t:.12g}",
    )
    for index in (1, 2, 3):
        point = next(
            p for p in additive.response_curve if abs(p.mass_fraction - index / 100) < 1e-9
        )
        values[f"gain_{index}"] = f"{point.cetane_gain:.12g}"
    return values


def scenario_from_editor(base: ScenarioConfig, values: dict[str, str]) -> ScenarioConfig:
    """Validate a detached synthetic scenario without touching preset files."""
    if base.mode.value != "model_demo" or base.controls:
        raise ValueError("Редактор доступен только для синтетической смеси")

    def number(key: str) -> float:
        try:
            value = float(values[key].strip().replace(",", "."))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{key}: введите число") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key}: требуется конечное неотрицательное число")
        return value

    data = base.model_dump(mode="json")
    data["id"] = base.id.removesuffix("_what_if") + "_what_if"
    data["assumptions"] = [*base.assumptions, "Synthetic what-if edited in UI; not plant data."]
    data["total_mass_t"] = number("total_mass_t")
    if data["total_mass_t"] <= 0:
        raise ValueError("Масса партии должна быть больше нуля")
    for component in data["blend_components"]:
        for field, _ in COMPONENT_FIELDS:
            key = f"{component['id']}.{field}"
            if field == "fraction":
                data["current_blend_mass_fractions"][component["id"]] = number(key) / 100
            elif "." in field:
                metric, bound = field.split(".")
                component[metric][bound] = None if not values[key].strip() else number(key)
                component[metric]["reference"] = "Synthetic what-if UI input"
                component[metric]["interval_kind"] = "scenario_bound"
                component[metric]["interval_level"] = None
            else:
                component[field] = number(key)
        for metric, bound in (("sulfur", "upper"), ("t95", "upper"), ("cetane_number", "lower")):
            estimate = component[metric]
            point, edge = estimate["value"], estimate[bound]
            if point is not None and edge is not None:
                if (bound == "upper" and edge < point) or (bound == "lower" and edge > point):
                    raise ValueError(f"{component['id']}: граница {metric} противоречит значению")
        if component["risk_index"] > 1:
            raise ValueError(f"{component['id']}: риск должен быть от 0 до 1")
    data["current_additive_mass_fraction"] = number("dose") / 100
    if abs(sum(data["current_blend_mass_fractions"].values()) + number("dose") / 100 - 1) > 1e-9:
        raise ValueError("Доли компонентов вместе с присадкой должны составлять 100%")
    cap = number("max_dose")
    if cap not in (1, 2, 3) or number("dose") > cap:
        raise ValueError("Максимум присадки: 1, 2 или 3%; текущая доза не выше максимума")
    gains = [0.0, *(number(f"gain_{i}") for i in (1, 2, 3))]
    if any(b < a for a, b in zip(gains, gains[1:], strict=False)):
        raise ValueError("Прирост цетанового числа не должен убывать с дозой")
    data["cetane_additive"].update(
        max_mass_fraction=cap / 100,
        fraction_step=0.01,
        available_mass_t=number("additive_stock"),
        cost_proxy_per_t=number("additive_cost"),
        response_curve=[
            {"mass_fraction": i / 100, "cetane_gain": gain} for i, gain in enumerate(gains)
        ],
        evidence_ref="Synthetic what-if UI input; not plant-calibrated",
    )
    return ScenarioConfig.model_validate(data)


def open_editor(
    parent: tk.Tk,
    preset: ScenarioConfig,
    current: ScenarioConfig,
    calculate: Callable[[ScenarioConfig], None],
) -> tk.Toplevel:
    dialog = tk.Toplevel(parent)
    dialog.title("Синтетический what-if — не реальные данные")
    dialog.geometry("940x680")
    dialog.minsize(800, 620)
    dialog.transient(parent)
    tk.Label(
        dialog,
        text="Синтетические свойства и запасы. Реальное управление отключено.",
        fg="#A55E00",
        font=("Segoe UI", 12, "bold"),
    ).pack(pady=(14, 6))
    tk.Label(
        dialog, text="Пустое свойство означает неизвестное, а не ноль. Доли с присадкой = 100%."
    ).pack()
    notebook = ttk.Notebook(dialog)
    notebook.pack(fill="both", expand=True, padx=16, pady=10)
    variables = {
        key: tk.StringVar(dialog, value=value) for key, value in editor_values(current).items()
    }
    components = ttk.Frame(notebook, padding=12)
    notebook.add(components, text="Компоненты и текущая рецептура")
    for col, component in enumerate(preset.blend_components):
        frame = ttk.LabelFrame(components, text=f"Компонент {component.id}", padding=10)
        frame.grid(row=0, column=col, sticky="nsew", padx=8)
        components.columnconfigure(col, weight=1)
        for row, (field, label) in enumerate(COMPONENT_FIELDS):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=variables[f"{component.id}.{field}"], width=14).grid(
                row=row, column=1, padx=10
            )
    additive = ttk.Frame(notebook, padding=20)
    notebook.add(additive, text="Партия и присадка")
    for row, (key, label) in enumerate(ADDITIVE_FIELDS):
        ttk.Label(additive, text=label).grid(row=row, column=0, sticky="w", pady=7)
        ttk.Entry(additive, textvariable=variables[key], width=20).grid(row=row, column=1, padx=16)
    error = tk.StringVar(dialog)
    tk.Label(dialog, textvariable=error, fg="#B3473C", wraplength=880, justify="left").pack(
        fill="x", padx=20
    )

    def apply() -> None:
        try:
            scenario = scenario_from_editor(
                preset, {key: var.get() for key, var in variables.items()}
            )
        except ValueError as exc:
            error.set(str(exc))
            return
        calculate(scenario)
        dialog.destroy()

    def reset() -> None:
        for key, value in editor_values(preset).items():
            variables[key].set(value)
        error.set(
            "Восстановлены исходные поля сценария. "
            "Нажмите «Пересчитать what-if» для нового результата."
        )

    buttons = ttk.Frame(dialog, padding=16)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Пересчитать what-if", command=apply).pack(side="left")
    ttk.Button(buttons, text="Сбросить к сценарию", command=reset).pack(side="left", padx=10)
    ttk.Button(buttons, text="Закрыть без изменений", command=dialog.destroy).pack(side="right")
    return dialog

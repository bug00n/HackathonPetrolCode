"""Synthetic-only editor; use the existing scenario contracts and decision engine."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from functools import partial
from tkinter import ttk

from source.agents.blend_assist import BlendOption, assist_blend
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


def assist_editor_values(
    base: ScenarioConfig, values: dict[str, str], locked: frozenset[str]
) -> tuple[tuple[BlendOption, dict[str, str]], ...]:
    """Parse physical inputs, then repair only the freely adjustable recipe fields."""
    if len(base.blend_components) != 2:
        raise ValueError("Автоподбор поддерживает два компонента")
    component_ids = tuple(item.id for item in base.blend_components)

    def fraction(key: str) -> float:
        try:
            result = float(values[key].strip().replace(",", ".")) / 100
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{key}: введите число от 0 до 100") from exc
        if not math.isfinite(result) or result < 0 or result > 1:
            raise ValueError(f"{key}: требуется число от 0 до 100")
        return result

    desired = {key: fraction(f"{key}.fraction") for key in component_ids}
    dose = fraction("dose")
    temporary = dict(values)
    # Only the recipe is temporarily balanced so scenario_from_editor can validate
    # unchanged component properties before the actual constrained search.
    try:
        max_dose = float(values["max_dose"].strip().replace(",", ".")) / 100
    except (KeyError, ValueError) as exc:
        raise ValueError("max_dose: введите максимум присадки 1, 2 или 3%") from exc
    temporary_dose = min(dose, max_dose) if math.isfinite(max_dose) else dose
    temporary["dose"] = f"{temporary_dose * 100:.12g}"
    temporary[f"{component_ids[0]}.fraction"] = f"{(1 - temporary_dose) * 50:.12g}"
    temporary[f"{component_ids[1]}.fraction"] = f"{(1 - temporary_dose) * 50:.12g}"
    scenario = scenario_from_editor(base, temporary)
    fixed = frozenset(
        key.removesuffix(".fraction") for key in locked if key.endswith(".fraction")
    ) | (frozenset({"dose"}) if "dose" in locked else frozenset())
    options = assist_blend(scenario, desired, dose, fixed)
    if not options:
        mass = scenario.total_mass_t or 0
        for component in scenario.blend_components:
            if component.id in fixed and desired[component.id] * mass > component.available_mass_t:
                needed = desired[component.id] * mass
                raise ValueError(
                    f"Компонента {component.id} нужно {needed:.3g} т, "
                    f"доступно {component.available_mass_t:.3g} т. "
                    "Снимите фиксацию или измените долю."
                )
        raise ValueError(
            "Подходящей смеси нет при заданных свойствах, запасах и закреплённых долях. "
            "Снимите фиксацию или измените указанное свойство или массу партии."
        )
    result = []
    for option in options:
        updated = dict(values)
        for key, share in option.fractions.items():
            updated[f"{key}.fraction"] = f"{share * 100:.12g}"
        updated["dose"] = f"{option.dose * 100:.12g}"
        scenario_from_editor(base, updated)
        result.append((option, updated))
    return tuple(result)


def open_editor(
    parent: tk.Tk,
    preset: ScenarioConfig,
    current: ScenarioConfig,
    calculate: Callable[[ScenarioConfig], None],
    *,
    calculate_with_context: Callable[[ScenarioConfig, dict[str, object]], None] | None = None,
) -> tk.Toplevel:
    dialog = tk.Toplevel(parent)
    dialog.title("Синтетический what-if — не реальные данные")
    dialog.geometry("940x700")
    dialog.minsize(800, 700)
    dialog.transient(parent)
    dialog.configure(bg="#F5F7F7")
    style = ttk.Style(dialog)
    style.configure("WhatIf.TNotebook", background="#F5F7F7", borderwidth=0)
    style.configure(
        "WhatIf.TNotebook.Tab",
        background="#EDF1F3",
        foreground="#38485A",
        padding=(16, 7),
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "WhatIf.TNotebook.Tab",
        background=[("selected", "#FFFFFF")],
        foreground=[("selected", "#007D78")],
    )
    style.configure("WhatIf.TFrame", background="#FFFFFF")
    style.configure("WhatIf.Footer.TFrame", background="#F5F7F7")
    style.configure("WhatIf.TLabelframe", background="#FFFFFF", bordercolor="#D8E0E4")
    style.configure(
        "WhatIf.TLabelframe.Label",
        background="#FFFFFF",
        foreground="#101B28",
        font=("Segoe UI", 11, "bold"),
    )
    style.configure(
        "WhatIf.TLabel", background="#FFFFFF", foreground="#38485A", font=("Segoe UI", 10)
    )
    style.configure("WhatIf.TEntry", fieldbackground="#FFFFFF", foreground="#101B28", padding=3)
    style.configure(
        "WhatIf.TButton",
        background="#FFFFFF",
        foreground="#101B28",
        bordercolor="#D8E0E4",
        padding=(16, 9),
        font=("Segoe UI", 10),
    )
    style.map("WhatIf.TButton", background=[("active", "#EDF1F3")])
    style.configure(
        "WhatIf.Primary.TButton",
        background="#007D78",
        foreground="#FFFFFF",
        bordercolor="#007D78",
        padding=(18, 9),
        font=("Segoe UI", 10, "bold"),
    )
    style.map("WhatIf.Primary.TButton", background=[("active", "#006965")])
    heading = tk.Frame(dialog, bg="#F5F7F7")
    heading.pack(fill="x", padx=24, pady=(10, 6))
    tk.Label(
        heading,
        text="Синтетический what-if",
        bg="#F5F7F7",
        fg="#101B28",
        font=("Segoe UI", 18, "bold"),
    ).pack(anchor="w")
    tk.Label(
        heading,
        text="Реальное управление отключено. Доли компонентов и присадки должны составлять 100 %.",
        bg="#F5F7F7",
        fg="#617082",
        font=("Segoe UI", 10),
    ).pack(anchor="w", pady=(2, 0))
    warning = tk.Label(
        dialog,
        text="Пустое свойство означает неизвестное значение, а не ноль.",
        bg="#FFF8E8",
        fg="#8C5400",
        anchor="w",
        font=("Segoe UI", 10, "bold"),
        padx=16,
        pady=6,
        highlightbackground="#E9B850",
        highlightthickness=1,
    )
    warning.pack(fill="x", padx=24, pady=(0, 8))
    notebook = ttk.Notebook(dialog, style="WhatIf.TNotebook")
    notebook.pack(fill="both", expand=True, padx=24, pady=(0, 8))
    variables = {
        key: tk.StringVar(dialog, value=value) for key, value in editor_values(current).items()
    }
    recipe = ttk.Frame(notebook, padding=20, style="WhatIf.TFrame")
    notebook.add(recipe, text="Рецептура")
    ttk.Label(
        recipe,
        text="Введите значение. Свободные доли подберутся после Enter или выхода из поля.",
        style="WhatIf.TLabel",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))
    locks: dict[str, tk.BooleanVar] = {}
    entries: dict[str, ttk.Entry] = {}
    recipe_keys = [f"{item.id}.fraction" for item in preset.blend_components] + ["dose"]
    for row, key in enumerate(recipe_keys, 1):
        label = "Присадка, %" if key == "dose" else f"Компонент {key.split('.')[0]}, %"
        ttk.Label(recipe, text=label, style="WhatIf.TLabel").grid(
            row=row, column=0, sticky="w", pady=7
        )
        entry = ttk.Entry(recipe, textvariable=variables[key], width=18, style="WhatIf.TEntry")
        entry.grid(row=row, column=1, padx=12, sticky="w")
        entries[key] = entry
        locks[key] = tk.BooleanVar(dialog, value=False)
        ttk.Checkbutton(recipe, text="Зафиксировать", variable=locks[key]).grid(
            row=row, column=2, sticky="w"
        )
    ttk.Label(recipe, text="Масса партии, т", style="WhatIf.TLabel").grid(
        row=4, column=0, sticky="w", pady=7
    )
    entries["total_mass_t"] = ttk.Entry(
        recipe, textvariable=variables["total_mass_t"], width=18, style="WhatIf.TEntry"
    )
    entries["total_mass_t"].grid(row=4, column=1, padx=12, sticky="w")
    preview = tk.StringVar(
        dialog, value="Исходный сценарий. Измените поле или нажмите «Подобрать смесь»."
    )
    ttk.Label(
        recipe, textvariable=preview, style="WhatIf.TLabel", wraplength=790, justify="left"
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(18, 6))
    variants = ttk.Combobox(recipe, state="readonly", width=38)
    variants.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)
    variants.grid_remove()
    components = ttk.Frame(notebook, padding=10, style="WhatIf.TFrame")
    notebook.add(components, text="Свойства и запасы")
    for col, component in enumerate(preset.blend_components):
        frame = ttk.LabelFrame(
            components,
            text=f"Компонент {component.id}",
            padding=8,
            style="WhatIf.TLabelframe",
        )
        frame.grid(row=0, column=col, sticky="nsew", padx=8)
        components.columnconfigure(col, weight=1)
        for row, (field, label) in enumerate(COMPONENT_FIELDS):
            if field == "fraction":
                continue
            ttk.Label(frame, text=label, style="WhatIf.TLabel").grid(
                row=row, column=0, sticky="w", pady=2
            )
            entry = ttk.Entry(
                frame,
                textvariable=variables[f"{component.id}.{field}"],
                width=14,
                style="WhatIf.TEntry",
            )
            entry.grid(row=row, column=1, padx=10)
            entries[f"{component.id}.{field}"] = entry
    additive = ttk.Frame(notebook, padding=20, style="WhatIf.TFrame")
    notebook.add(additive, text="Параметры присадки")
    for row, (key, label) in enumerate(ADDITIVE_FIELDS):
        if key in {"total_mass_t", "dose"}:
            continue
        ttk.Label(additive, text=label, style="WhatIf.TLabel").grid(
            row=row, column=0, sticky="w", pady=7
        )
        entry = ttk.Entry(additive, textvariable=variables[key], width=20, style="WhatIf.TEntry")
        entry.grid(row=row, column=1, padx=16)
        entries[key] = entry
    error = tk.StringVar(dialog)
    tk.Label(
        dialog,
        textvariable=error,
        bg="#F5F7F7",
        fg="#B3473C",
        wraplength=880,
        justify="left",
    ).pack(fill="x", padx=24)

    style.configure("WhatIf.Changed.TEntry", fieldbackground="#E2F6EF", padding=3)
    assistant_on = tk.BooleanVar(dialog, value=True)
    before_assist: dict[str, str] | None = None
    options: tuple[tuple[BlendOption, dict[str, str]], ...] = ()
    pending: str | None = None
    updating = False

    def values_now() -> dict[str, str]:
        return {key: var.get() for key, var in variables.items()}

    def show_option(position: int) -> None:
        nonlocal updating
        option, updated = options[position]
        old = values_now()
        updating = True
        try:
            for key in recipe_keys:
                variables[key].set(updated[key])
                entries[key].configure(
                    style=("WhatIf.Changed.TEntry" if updated[key] != old[key] else "WhatIf.TEntry")
                )
        finally:
            updating = False
        names = {
            **{f"{item.id}.fraction": f"Компонент {item.id}" for item in preset.blend_components},
            "dose": "Присадка",
        }
        changes = [
            f"{names[key]}: {old[key]}% → {updated[key]}%"
            for key in recipe_keys
            if old[key] != updated[key]
        ]
        quality = next(
            (item.metrics for item in option.evaluation.assessments if "sulfur" in item.metrics),
            {},
        )
        checks = []
        for constraint in preset.constraints:
            metric = quality.get(constraint.metric)
            if metric is None:
                continue
            bound = (
                "upper"
                if constraint.use_upper_estimate
                else "lower"
                if constraint.use_lower_estimate
                else "value"
            )
            value = getattr(metric, bound)
            limit = constraint.upper if constraint.upper is not None else constraint.lower
            if value is not None and limit is not None:
                sign = "≤" if constraint.upper is not None else "≥"
                label = {"sulfur": "Сера", "t95": "T95", "cetane_number": "ЦЧ"}.get(
                    constraint.metric, constraint.metric
                )
                checks.append(f"{label}: {value:.2f} {sign} {limit:g}")
        preview.set(
            f"{option.label}. "
            + ("; ".join(changes) if changes else "Смесь уже подходит.")
            + ("\nПроверка: " + "; ".join(checks) if checks else "")
        )
        error.set("")

    def pick(_: tk.Event[tk.Misc] | None = None) -> None:
        if variants.current() >= 0:
            show_option(variants.current())

    variants.bind("<<ComboboxSelected>>", pick)

    def suggest(changed: str | None = None) -> None:
        nonlocal options, before_assist, pending
        pending = None
        if updating or not assistant_on.get():
            return
        raw = values_now()
        fixed = frozenset(key for key, locked in locks.items() if locked.get())
        if changed in locks:
            fixed |= {changed}
        try:
            options = assist_editor_values(preset, raw, fixed)
        except ValueError as exc:
            options = ()
            variants.grid_remove()
            error.set(str(exc))
            preview.set("Автоподбор не смог составить допустимую смесь.")
            return
        before_assist = raw
        variants["values"] = tuple(item.label for item, _ in options)
        variants.current(0)
        variants.grid()
        show_option(0)

    def schedule(changed: str) -> None:
        nonlocal pending
        if pending is not None:
            dialog.after_cancel(pending)
        pending = dialog.after(400, lambda: suggest(changed))

    def on_return(_event: tk.Event[tk.Misc], *, name: str) -> None:
        suggest(name)

    def on_focus_out(_event: tk.Event[tk.Misc], *, name: str) -> None:
        schedule(name)

    for key, entry in entries.items():
        entry.bind("<Return>", partial(on_return, name=key))
        entry.bind("<FocusOut>", partial(on_focus_out, name=key))

    def undo_assist() -> None:
        nonlocal updating, before_assist
        if before_assist is None:
            return
        updating = True
        try:
            for key in recipe_keys:
                if key not in locks or not locks[key].get():
                    variables[key].set(before_assist[key])
                entries[key].configure(style="WhatIf.TEntry")
        finally:
            updating = False
        before_assist = None
        preview.set("Автоподбор отменён. Введённое значение сохранено.")
        variants.grid_remove()

    ttk.Checkbutton(recipe, text="Автоподбор", variable=assistant_on).grid(
        row=7, column=0, sticky="w", pady=(12, 0)
    )
    ttk.Button(recipe, text="Подобрать смесь", command=suggest).grid(
        row=7, column=1, sticky="w", pady=(12, 0)
    )
    ttk.Button(recipe, text="Отменить автоподбор", command=undo_assist).grid(
        row=7, column=2, sticky="w", pady=(12, 0)
    )

    def apply() -> None:
        try:
            scenario = scenario_from_editor(
                preset, {key: var.get() for key, var in variables.items()}
            )
        except ValueError as exc:
            error.set(str(exc))
            return
        if calculate_with_context is None:
            calculate(scenario)
        else:
            calculate_with_context(
                scenario,
                {
                    "original_values": before_assist,
                    "displayed_values": values_now(),
                    "locked_fields": [key for key, value in locks.items() if value.get()],
                    "assisted": before_assist is not None,
                },
            )
        dialog.destroy()

    def reset() -> None:
        nonlocal before_assist
        for key, value in editor_values(preset).items():
            variables[key].set(value)
        for lock_var in locks.values():
            lock_var.set(False)
        before_assist = None
        variants.grid_remove()
        error.set(
            "Восстановлены исходные поля сценария. "
            "Нажмите «Пересчитать what-if» для нового результата."
        )

    buttons = ttk.Frame(dialog, padding=(24, 8), style="WhatIf.Footer.TFrame")
    buttons.pack(fill="x")
    ttk.Button(
        buttons, text="Пересчитать what-if", command=apply, style="WhatIf.Primary.TButton"
    ).pack(side="left")
    ttk.Button(buttons, text="Сбросить к сценарию", command=reset, style="WhatIf.TButton").pack(
        side="left", padx=10
    )
    ttk.Button(
        buttons, text="Закрыть без изменений", command=dialog.destroy, style="WhatIf.TButton"
    ).pack(side="right")
    return dialog

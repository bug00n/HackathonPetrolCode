"""Native Tk history chart with real timestamps and explicit missing-data gaps."""

from __future__ import annotations

import math
import tkinter as tk
from datetime import datetime, timedelta
from typing import Any

ForecastPoint = tuple[datetime, float | None, float | None]


def history_points(payload: dict[str, Any] | None) -> list[ForecastPoint]:
    if not payload:
        return []
    if isinstance(payload.get("results"), list):
        rows = payload["results"]
    else:
        carrier = payload.get("selected") or payload.get("baseline") or {}
        metric: dict[str, Any] = next(
            (
                a.get("metrics", {}).get("sulfur")
                for a in carrier.get("assessments", [])
                if a.get("metrics", {}).get("sulfur") is not None
            ),
            {},
        )
        rows = [
            {
                "as_of": payload.get("as_of"),
                "sulfur": {"point": metric.get("value"), "upper": metric.get("upper")},
            }
        ]
    if not rows:
        return []
    timezone = datetime.fromisoformat(
        str(payload.get("start") or payload.get("as_of") or rows[0]["as_of"])
    ).tzinfo

    def finite(value: object) -> float | None:
        return (
            float(value)
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            else None
        )

    points = []
    for row in rows:
        if not row.get("as_of"):
            continue
        at = datetime.fromisoformat(row["as_of"])
        if at.tzinfo is None:
            raise ValueError("Время прогноза должно содержать часовой пояс")
        metric = row.get("sulfur") or {}
        points.append(
            (
                (at + timedelta(minutes=60)).astimezone(timezone),
                finite(metric.get("point")),
                finite(metric.get("upper")),
            )
        )
    return sorted(points, key=lambda point: point[0])


class HistoryChart(tk.Canvas):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, height=290, background="white", highlightthickness=0)
        self.points: list[ForecastPoint] = []
        self.bind("<Configure>", lambda _: self.draw())
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda _: self.itemconfigure("hover", text=""))

    def set_payload(self, payload: dict[str, Any] | None) -> None:
        self.points = history_points(payload)
        self.draw()

    def draw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 600), max(self.winfo_height(), 290)
        left, right, top, bottom = 62.0, float(width - 24), 52.0, float(height - 64)
        self.create_text(
            left,
            16,
            anchor="w",
            fill="#007D78",
            text="● Прогноз серы     ● Верхняя оценка (синяя)     — Предел 10 мг/кг",
        )
        self.create_text(
            right,
            height - 10,
            anchor="e",
            fill="#617082",
            text="Время прогноза = момент данных + 60 мин; наведите мышь на график",
        )
        values = [
            value
            for _, point, upper in self.points
            for value in (point, upper)
            if value is not None
        ]
        if not values:
            self.create_text(
                width / 2,
                height / 2,
                text="Нет численного прогноза. Укажите время и выполните расчёт.",
                fill="#617082",
            )
            return
        minimum, maximum = min(0.0, min(values)), max(10.0, max(values)) * 1.12
        start, end = self.points[0][0].timestamp(), self.points[-1][0].timestamp()

        def x(at: datetime) -> float:
            return (
                (left + right) / 2
                if start == end
                else left + (at.timestamp() - start) / (end - start) * (right - left)
            )

        def y(value: float) -> float:
            return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

        for i in range(5):
            value = minimum + (maximum - minimum) * i / 4
            self.create_line(left, y(value), right, y(value), fill="#E3E8EA")
            self.create_text(left - 8, y(value), text=f"{value:.1f}", anchor="e", fill="#617082")
        self.create_text(8, 38, anchor="w", text="мг/кг", fill="#617082")
        self.create_line(
            left, y(10), right, y(10), fill="#D88400", dash=(6, 4), width=2, tags="limit"
        )
        for series, color, tag in ((1, "#007D78", "point"), (2, "#3366BB", "upper")):
            segment: list[float] = []

            def flush(segment: list[float] = segment, color: str = color, tag: str = tag) -> None:
                if len(segment) >= 4:
                    self.create_line(*segment, fill=color, width=2, tags=tag)
                elif len(segment) == 2:
                    px, py = segment
                    self.create_oval(
                        px - 3, py - 3, px + 3, py + 3, fill=color, outline=color, tags=tag
                    )
                segment.clear()

            for item in self.points:
                series_value = item[1] if series == 1 else item[2]
                if series_value is None:
                    flush()
                    continue
                px, py = x(item[0]), y(series_value)
                segment.extend((px, py))
                if len(self.points) <= 60:
                    self.create_oval(
                        px - 3, py - 3, px + 3, py + 3, fill=color, outline=color, tags=tag
                    )
            flush()
        indices = sorted({round(i * (len(self.points) - 1) / 4) for i in range(5)})
        for index in indices:
            at = self.points[index][0]
            self.create_text(
                x(at),
                bottom + 24,
                text=at.strftime("%d.%m.%y\n%H:%M %z"),
                anchor="w" if index == 0 else "e" if index == len(self.points) - 1 else "center",
                fill="#617082",
                font=("Segoe UI", 8),
            )
        self.create_text(
            left,
            35,
            anchor="w",
            text="",
            tags="hover",
            fill="#101B28",
            font=("Segoe UI", 9, "bold"),
        )

    def _hover(self, event: tk.Event[Any]) -> None:
        if not self.points or not self.find_withtag("hover"):
            return
        start, end = self.points[0][0].timestamp(), self.points[-1][0].timestamp()
        fraction = min(1.0, max(0.0, (event.x - 62) / max(1, self.winfo_width() - 86)))
        point = min(
            self.points, key=lambda p: abs(p[0].timestamp() - (start + fraction * (end - start)))
        )
        value = "нет данных" if point[1] is None else f"{point[1]:.4f}"
        upper = "нет данных" if point[2] is None else f"{point[2]:.4f}"
        self.itemconfigure(
            "hover", text=f"{point[0]:%d.%m.%Y %H:%M %z}   прогноз {value}; верхняя {upper} мг/кг"
        )

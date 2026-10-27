"""Shared split-aware summaries used by training and experiment reports."""
from __future__ import annotations

import statistics


def group_macro_f1(details: list[dict], rows: list[dict]) -> tuple[float, dict[str, float]]:
    """Average RNA F1 within each isolation group, then equally across groups."""
    group_by_id = {row["id"]: row["split_group_id"] for row in rows}
    values: dict[str, list[float]] = {}
    for detail in details:
        if detail["id"] not in group_by_id:
            raise ValueError(f"Evaluation detail lacks a corresponding group: {detail['id']}")
        values.setdefault(group_by_id[detail["id"]], []).append(float(detail["f1"]))
    if not values:
        raise ValueError("Cannot compute group-macro F1 without evaluated RNAs")
    per_group = {group: statistics.mean(scores) for group, scores in values.items()}
    return statistics.mean(per_group.values()), per_group

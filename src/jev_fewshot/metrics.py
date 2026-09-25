"""Multiclass metrics and a deterministic hierarchical paired bootstrap."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from . import LABELS


def accuracy(actual: Sequence[str], predicted: Sequence[str]) -> float:
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual) if actual else 0.0


def macro_f1(actual: Sequence[str], predicted: Sequence[str]) -> float:
    values = []
    for label in LABELS:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(values) / len(values)


def per_class_recall(actual: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    return {label: (sum(a == label and p == label for a, p in zip(actual, predicted)) /
                    sum(a == label for a in actual) if any(a == label for a in actual) else 0.0)
            for label in LABELS}


def _p(row: dict, label: str) -> float:
    return float(row.get("probabilities", {}).get(label, 0.0))


def multiclass_log_loss(rows: Sequence[dict]) -> float:
    return -sum(math.log(max(_p(row, row["actual"]), 1e-12)) for row in rows) / len(rows) if rows else 0.0


def multiclass_brier(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    return sum(sum((_p(row, label) - float(row["actual"] == label)) ** 2 for label in LABELS)
               for row in rows) / len(rows)


def top_label_ece(rows: Sequence[dict], bins: int = 10) -> float:
    buckets: list[list[dict]] = [[] for _ in range(bins)]
    for row in rows:
        confidence = max((_p(row, label) for label in LABELS), default=0.0)
        buckets[min(int(confidence * bins), bins - 1)].append(row)
    return sum(len(bucket) / len(rows) * abs(
        sum(max((_p(row, label) for label in LABELS), default=0.0) for row in bucket) / len(bucket)
        - sum(row["actual"] == row["prediction"] for row in bucket) / len(bucket))
        for bucket in buckets if bucket) if rows else 0.0


def summarize(rows: Sequence[dict]) -> dict:
    actual = [row["actual"] for row in rows]
    predicted = [row["prediction"] for row in rows]
    return {"n": len(rows), "accuracy": accuracy(actual, predicted), "macro_f1": macro_f1(actual, predicted),
            "log_loss": multiclass_log_loss(rows), "brier": multiclass_brier(rows),
            "ece": top_label_ece(rows), "recall": per_class_recall(actual, predicted),
            "mean_latency_ms": sum(row.get("latency_ms", 0.0) for row in rows) / len(rows) if rows else 0.0,
            "input_tokens": sum((row.get("usage") or {}).get("input_tokens") or 0 for row in rows),
            "output_tokens": sum((row.get("usage") or {}).get("output_tokens") or 0 for row in rows)}


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def hierarchical_accuracy_delta(zero_rows: Sequence[dict], few_rows: Sequence[dict], *,
                                resamples: int = 2000, seed: int = 20260925) -> Interval:
    """Bootstrap target IDs and demonstration draws, treating each as random sampling units."""
    zero = {row["target_id"]: row for row in zero_rows}
    by_draw: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in few_rows:
        by_draw[int(row["draw_seed"])][row["target_id"]] = row
    target_ids = sorted(set(zero).intersection(*(set(rows) for rows in by_draw.values()))) if by_draw else []
    draws = sorted(by_draw)
    if not target_ids or not draws:
        return Interval(0.0, 0.0, 0.0)
    def delta(draw_selection: Iterable[int], target_selection: Iterable[str]) -> float:
        differences = []
        for draw in draw_selection:
            for target_id in target_selection:
                differences.append(float(by_draw[draw][target_id]["prediction"] == zero[target_id]["actual"])
                                   - float(zero[target_id]["prediction"] == zero[target_id]["actual"]))
        return sum(differences) / len(differences)
    estimate = delta(draws, target_ids)
    rng = random.Random(seed)
    samples = [delta([rng.choice(draws) for _ in draws],
                     [rng.choice(target_ids) for _ in target_ids]) for _ in range(resamples)]
    samples.sort()
    return Interval(estimate, samples[int(0.025 * resamples)], samples[int(0.975 * resamples)])

"""Dataset loading and deterministic, label-safe experiment selections."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import DATASET_NAME, DATASET_REVISION, DRAW_SEEDS, LABELS, TEST_PER_CLASS, TEST_SEED


@dataclass(frozen=True)
class Article:
    source_split: str
    source_index: int
    text: str
    label: str

    @property
    def manifest_id(self) -> str:
        return f"{self.source_split}-{self.source_index}"


def normalized_text(text: str) -> str:
    """The comparison key used solely to prevent exact train/test leakage."""
    return re.sub(r"\s+", " ", text.strip()).casefold()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def load_ag_news(*, cache_dir: str | Path | None = None) -> tuple[list[Article], list[Article]]:
    """Download the pinned upstream revision; no corpus rows are shipped here."""
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - install-time guidance
        raise RuntimeError("Install dependencies with `make install` first.") from error
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, cache_dir=str(cache_dir) if cache_dir else None)

    def records(split: str) -> list[Article]:
        return [Article(split, i, row["text"], LABELS[int(row["label"])])
                for i, row in enumerate(dataset[split])]
    return records("train"), records("test")


def balanced_scoreboard(test: Sequence[Article], *, per_class: int = TEST_PER_CLASS,
                        seed: int = TEST_SEED) -> list[Article]:
    """Choose a fixed stratified sample without examining article text or predictions."""
    rng = random.Random(seed)
    selected: list[Article] = []
    for label in LABELS:
        candidates = [row for row in test if row.label == label]
        if len(candidates) < per_class:
            raise ValueError(f"Only {len(candidates)} test rows for {label}")
        selected.extend(rng.sample(candidates, per_class))
    return sorted(selected, key=lambda row: (LABELS.index(row.label), row.source_index))


def demonstration_draws(train: Sequence[Article], scoreboard: Sequence[Article],
                        *, seeds: Iterable[int] = DRAW_SEEDS) -> dict[int, list[Article]]:
    """Return four leakage-safe examples per class per seed; shot levels are prefixes."""
    held_out_hashes = {text_hash(row.text) for row in scoreboard}
    draws: dict[int, list[Article]] = {}
    for seed in seeds:
        rng = random.Random(seed)
        draw: list[Article] = []
        for label in LABELS:
            candidates = [row for row in train
                          if row.label == label and text_hash(row.text) not in held_out_hashes]
            if len(candidates) < 4:
                raise ValueError(f"Not enough safe train examples for {label}")
            draw.extend(rng.sample(candidates, 4))
        # Phase I deliberately preserves one canonical display order: World, Sports,
        # Business, Sci/Tech. Ordering sensitivity is a separate follow-up after
        # the primary result, not a variable silently mixed into it.
        draws[seed] = draw
    return draws


def examples_for_shots(draw: Sequence[Article], shots: int) -> list[Article]:
    """Return nested, class-balanced prefixes: 1/2/4 examples per class."""
    if shots == 0:
        return []
    if shots not in (4, 8, 16):
        raise ValueError("shots must be 0, 4, 8, or 16")
    per_class = shots // len(LABELS)
    selected: list[Article] = []
    for label in LABELS:
        members = [row for row in draw if row.label == label]
        selected.extend(members[:per_class])
    # Preserve the canonical class and within-class selection order after taking each prefix.
    ids = {row.manifest_id for row in selected}
    return [row for row in draw if row.manifest_id in ids]


def manifest_rows(scoreboard: Sequence[Article]) -> list[dict]:
    """The committed scoreboard format deliberately contains no article text."""
    return [{"id": row.manifest_id, "source_split": row.source_split,
             "source_index": row.source_index, "label": row.label, "text_sha256": text_hash(row.text)}
            for row in scoreboard]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

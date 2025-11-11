from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from poliastro.bodies import Body

DATASET_PATH = Path("pipeline_outputs/refinement_samples.jsonl")


def _ensure_dataset_path() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


def record_refinement_sample(
    departure_body: Body,
    arrival_body: Body,
    date_offset: int,
    tof_offset: int,
    delta_v_kms: float,
    success: bool,
) -> None:
    _ensure_dataset_path()
    entry = {
        "departure_body": departure_body.name,
        "arrival_body": arrival_body.name,
        "date_offset": int(date_offset),
        "tof_offset": int(tof_offset),
        "delta_v_kms": float(delta_v_kms),
        "success": bool(success),
    }
    with DATASET_PATH.open("a", encoding="utf-8") as dataset:
        dataset.write(json.dumps(entry))
        dataset.write("\n")


def suggest_refinement_offsets(
    departure_body: Body,
    arrival_body: Body,
    date_offsets: Iterable[int],
    tof_offsets: Iterable[int],
    max_suggestions: int = 12,
) -> List[Tuple[int, int]]:
    if not DATASET_PATH.exists():
        return []

    allowed_dates = set(date_offsets)
    allowed_tofs = set(tof_offsets)

    suggestions: List[Tuple[int, int, float]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as dataset:
        for line in dataset:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("success") is not True:
                continue
            if (
                record.get("departure_body") != departure_body.name
                or record.get("arrival_body") != arrival_body.name
            ):
                continue

            date_offset = int(record.get("date_offset", 0))
            tof_offset = int(record.get("tof_offset", 0))
            if date_offset not in allowed_dates or tof_offset not in allowed_tofs:
                continue
            delta_v = float(record.get("delta_v_kms", 0.0))
            suggestions.append((date_offset, tof_offset, delta_v))

    suggestions.sort(key=lambda item: item[2])
    return [(date, tof) for date, tof, _ in suggestions[:max_suggestions]]


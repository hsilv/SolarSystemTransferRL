from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from astropy import units as u
from poliastro.bodies import Earth, Jupiter, Saturn, Venus


@dataclass(frozen=True)
class SequenceSpec:
    """Definition for a multi-gravity assist planetary sequence."""

    id: str
    name: str
    bodies: Tuple[Any, ...]
    tof_bounds: Tuple[Tuple[u.Quantity, u.Quantity], ...]
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.bodies) < 2:
            raise ValueError("Sequence must contain at least two bodies.")
        if len(self.tof_bounds) != len(self.bodies) - 1:
            raise ValueError("Time-of-flight bounds must match the number of legs.")
        for lower, upper in self.tof_bounds:
            if lower <= 0 * u.day or upper <= 0 * u.day:
                raise ValueError("TOF bounds must be positive.")
            if lower >= upper:
                raise ValueError("Lower TOF bound must be less than upper bound.")


def _days(min_days: float, max_days: float) -> Tuple[u.Quantity, u.Quantity]:
    return min_days * u.day, max_days * u.day


def sequence_catalog() -> Dict[str, SequenceSpec]:
    """Return the predefined catalog of planetary transfer sequences."""
    return {
        "E-S": SequenceSpec(
            id="E-S",
            name="Earth → Saturn",
            bodies=(Earth, Saturn),
            tof_bounds=(_days(2200, 3800),),
            description="Direct transfer to Saturn.",
        ),
        "E-J-S": SequenceSpec(
            id="E-J-S",
            name="Earth → Jupiter → Saturn",
            bodies=(Earth, Jupiter, Saturn),
            tof_bounds=(_days(500, 1100), _days(700, 1800)),
            description="Single gravity assist at Jupiter.",
        ),
        "E-V-J-S": SequenceSpec(
            id="E-V-J-S",
            name="Earth → Venus → Jupiter → Saturn",
            bodies=(Earth, Venus, Jupiter, Saturn),
            tof_bounds=(
                _days(100, 180),
                _days(250, 500),
                _days(700, 1600),
            ),
            description="Inner planet assist before heading to Jupiter.",
        ),
        "E-V-E-J-S": SequenceSpec(
            id="E-V-E-J-S",
            name="Earth → Venus → Earth → Jupiter → Saturn",
            bodies=(Earth, Venus, Earth, Jupiter, Saturn),
            tof_bounds=(
                _days(90, 150),
                _days(250, 420),
                _days(600, 900),
                _days(700, 1500),
            ),
            description="Double inner assists to set up a Jupiter sling to Saturn.",
        ),
    }


def get_sequence(sequence_id: str) -> SequenceSpec:
    catalog = sequence_catalog()
    try:
        return catalog[sequence_id]
    except KeyError as exc:
        raise KeyError(f"Sequence '{sequence_id}' is not defined in the catalog.") from exc


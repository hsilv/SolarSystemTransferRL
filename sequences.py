from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from astropy.time import Time
from poliastro.bodies import Body, Earth, Jupiter, Saturn, Venus

SequenceBounds = Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class SequenceDefinition:
    """Definition of a planetary encounter sequence and time-of-flight bounds."""

    identifier: str
    bodies: Tuple[Body, ...]
    tof_bounds_days: SequenceBounds
    reference_departure: Time

    def __post_init__(self) -> None:
        if len(self.bodies) < 2:
            raise ValueError("`bodies` must contain at least two entries.")
        if len(self.tof_bounds_days) != len(self.bodies) - 1:
            raise ValueError(
                "`tof_bounds_days` must contain one (min, max) pair per transfer leg."
            )
        for lower, upper in self.tof_bounds_days:
            if lower <= 0 or upper <= 0:
                raise ValueError("Time-of-flight bounds must be positive.")
            if lower >= upper:
                raise ValueError("Each lower bound must be strictly less than the upper bound.")


SEQUENCE_CATALOG: Tuple[SequenceDefinition, ...] = (
    SequenceDefinition(
        identifier="E-S",
        bodies=(Earth, Saturn),
        tof_bounds_days=((2200.0, 3800.0),),
        reference_departure=Time("2031-07-14", scale="tdb"),
    ),
    SequenceDefinition(
        identifier="E-J-S",
        bodies=(Earth, Jupiter, Saturn),
        tof_bounds_days=((500.0, 1100.0), (700.0, 1800.0)),
        reference_departure=Time("2031-07-14", scale="tdb"),
    ),
    SequenceDefinition(
        identifier="E-V-J-S",
        bodies=(Earth, Venus, Jupiter, Saturn),
        tof_bounds_days=((100.0, 180.0), (250.0, 500.0), (700.0, 1600.0)),
        reference_departure=Time("2031-07-14", scale="tdb"),
    ),
    SequenceDefinition(
        identifier="E-V-E-J-S",
        bodies=(Earth, Venus, Earth, Jupiter, Saturn),
        tof_bounds_days=(
            (90.0, 150.0),
            (250.0, 420.0),
            (600.0, 900.0),
            (700.0, 1500.0),
        ),
        reference_departure=Time("2031-07-14", scale="tdb"),
    ),
)


def get_sequence(identifier: str) -> SequenceDefinition:
    """Return the sequence definition matching `identifier`."""
    for sequence in SEQUENCE_CATALOG:
        if sequence.identifier == identifier:
            return sequence
    raise ValueError(f"Unknown sequence identifier: {identifier}")


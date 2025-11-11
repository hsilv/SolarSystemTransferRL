from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from astropy.time import Time
from poliastro.bodies import (
    Body,
    Earth,
    Jupiter,
    Saturn,
    Venus,
    Mercury,
    Mars,
    Uranus,
    Neptune,
)

SequenceBounds = Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class SequenceDefinition:
    identifier: str
    bodies: Tuple[Body, ...]
    tof_bounds_days: SequenceBounds
    reference_departure: Time

    def __post_init__(self) -> None:
        if len(self.bodies) < 2:
            raise ValueError("`bodies` must contain at least two entries.")
        if len(self.tof_bounds_days) != len(self.bodies) - 1:
            raise ValueError("`tof_bounds_days` must contain one (min, max) pair per transfer leg.")
        for lower, upper in self.tof_bounds_days:
            if lower <= 0 or upper <= 0:
                raise ValueError("Time-of-flight bounds must be positive.")
            if lower >= upper:
                raise ValueError("Each lower bound must be strictly less than the upper bound.")


PLANET_ORDER = (
    Mercury,
    Venus,
    Earth,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
)

PLANET_CODES = {
    Mercury: "M",
    Venus: "V",
    Earth: "E",
    Mars: "Ma",
    Jupiter: "J",
    Saturn: "S",
    Uranus: "U",
    Neptune: "N",
}

ADJACENT_TOF_DAYS = {
    ("Mercury", "Venus"): (80.0, 140.0),
    ("Venus", "Earth"): (80.0, 150.0),
    ("Earth", "Mars"): (230.0, 420.0),
    ("Mars", "Jupiter"): (370.0, 650.0),
    ("Jupiter", "Saturn"): (600.0, 1200.0),
    ("Saturn", "Uranus"): (1700.0, 3200.0),
    ("Uranus", "Neptune"): (2300.0, 4200.0),
}

REFERENCE_DEPARTURE = Time("2031-07-14", scale="tdb")


def _adjacent_tof(body_a: Body, body_b: Body) -> Tuple[float, float]:
    key = (body_a.name, body_b.name)
    if key not in ADJACENT_TOF_DAYS:
        key = (body_b.name, body_a.name)
    if key not in ADJACENT_TOF_DAYS:
        raise ValueError(f"No TOF data for pair {body_a.name}-{body_b.name}")
    return ADJACENT_TOF_DAYS[key]


def _build_sequence_identifier(bodies: Tuple[Body, ...]) -> str:
    return "-".join(PLANET_CODES.get(body, body.name[:2]) for body in bodies)


def _generate_sequences() -> Tuple[SequenceDefinition, ...]:
    sequences = []
    count = len(PLANET_ORDER)
    for origin_index in range(count):
        for destination_index in range(count):
            if origin_index == destination_index:
                continue
            step = 1 if destination_index > origin_index else -1
            indices = range(origin_index, destination_index + step, step)
            bodies = tuple(PLANET_ORDER[i] for i in indices)
            tof_bounds = tuple(_adjacent_tof(bodies[i], bodies[i + 1]) for i in range(len(bodies) - 1))
            sequences.append(
                SequenceDefinition(
                    identifier=_build_sequence_identifier(bodies),
                    bodies=bodies,
                    tof_bounds_days=tof_bounds,
                    reference_departure=REFERENCE_DEPARTURE,
                )
            )
    # Remove duplicates if any (shouldn't, but safeguard)
    unique = {seq.identifier: seq for seq in sequences}
    return tuple(unique.values())


SEQUENCE_CATALOG: Tuple[SequenceDefinition, ...] = _generate_sequences()


def get_sequence(identifier: str) -> SequenceDefinition:
    """Return the sequence definition matching `identifier`."""
    for sequence in SEQUENCE_CATALOG:
        if sequence.identifier == identifier:
            return sequence
    raise ValueError(f"Unknown sequence identifier: {identifier}")


def resolve_body(name: str) -> Body:
    """Return the Solar System body matching the provided name."""
    normalized = name.strip().lower()
    registry = {
        body.name.lower(): body
        for body in (
            Mercury,
            Venus,
            Earth,
            Mars,
            Jupiter,
            Saturn,
            Uranus,
            Neptune,
        )
    }
    try:
        return registry[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown body '{name}'.") from exc


def sequences_between(origin_name: str, destination_name: str) -> Tuple[SequenceDefinition, ...]:
    """Return catalog sequences whose endpoints match the requested bodies."""
    origin = resolve_body(origin_name)
    destination = resolve_body(destination_name)
    matches = tuple(
        sequence
        for sequence in SEQUENCE_CATALOG
        if sequence.bodies[0] == origin and sequence.bodies[-1] == destination
    )
    return matches


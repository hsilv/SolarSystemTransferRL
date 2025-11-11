from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pygmo as pg
from astropy import units as u

from mga import evaluate_mga
from sequences import SEQUENCE_CATALOG, SequenceDefinition, get_sequence


class MgaOptimizationProblem:
    """PyGMO problem wrapper to minimize total Δv for a fixed sequence."""

    def __init__(
        self,
        sequence_def: SequenceDefinition,
        departure_offset_bounds: Tuple[float, float],
        penalty: float = 1e6,
    ):
        self._sequence = sequence_def
        self._departure_offset_bounds = departure_offset_bounds
        self._penalty = penalty

    def fitness(self, x):
        departure_offset = x[0]
        tofs = x[1:]
        departure_date = self._sequence.reference_departure + departure_offset * u.day
        try:
            result = evaluate_mga(
                departure_date,
                self._sequence.bodies,
                tofs[: len(self._sequence.tof_bounds_days)],
            )
            return [result.total_delta_v.to_value(u.km / u.s)]
        except Exception:
            return [self._penalty]

    def get_bounds(self):
        lower = [self._departure_offset_bounds[0]]
        upper = [self._departure_offset_bounds[1]]
        for bounds in self._sequence.tof_bounds_days:
            lower.append(bounds[0])
            upper.append(bounds[1])
        return (lower, upper)

    def get_nobj(self):
        return 1

    def get_name(self):
        return f"MGA Δv minimization for {self._sequence.identifier}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyGMO benchmark for MGA trajectories.")
    parser.add_argument(
        "--sequence-id",
        type=str,
        default="E-J-S",
        help="Identifier of the sequence to optimize.",
    )
    parser.add_argument("--population", type=int, default=64, help="Population size.")
    parser.add_argument("--generations", type=int, default=200, help="Number of DE generations.")
    parser.add_argument(
        "--departure-offset-bounds",
        type=float,
        nargs=2,
        default=(-365.0, 365.0),
        metavar=("MIN", "MAX"),
        help="Bounds for departure date offset in days.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pygmo_best.json"),
        help="Path to save the best trajectory summary.",
    )
    return parser.parse_args()


def optimize_sequence(
    sequence_id: str,
    population: int = 64,
    generations: int = 200,
    departure_offset_bounds: Tuple[float, float] = (-365.0, 365.0),
    output: Path | str | None = None,
) -> Dict[str, object]:
    sequence_def = get_sequence(sequence_id)

    problem = pg.problem(
        MgaOptimizationProblem(
            sequence_def=sequence_def,
            departure_offset_bounds=departure_offset_bounds,
        )
    )

    algorithm = pg.algorithm(pg.de(gen=generations))
    population_obj = pg.population(problem, size=population)
    population_obj = algorithm.evolve(population_obj)

    best_x = population_obj.champion_x
    departure_offset = best_x[0]
    tof_values = best_x[1 : 1 + len(sequence_def.tof_bounds_days)]

    departure_date = sequence_def.reference_departure + departure_offset * u.day
    result = evaluate_mga(departure_date, sequence_def.bodies, tof_values)

    summary = {
        "sequence_id": sequence_def.identifier,
        "departure_date_tdb": result.departure_date.tdb.iso,
        "arrival_date_tdb": result.legs[-1].summary.arrival_date.tdb.iso,
        "departure_offset_days": float(departure_offset),
        "time_of_flights_days": [float(v) for v in tof_values],
        "delta_v_total_kms": result.total_delta_v.to_value(u.km / u.s),
        "delta_v_initial_kms": result.initial_delta_v.to_value(u.km / u.s),
        "delta_v_arrival_kms": result.arrival_delta_v.to_value(u.km / u.s),
        "delta_v_intermediate_kms": [
            dv.to_value(u.km / u.s) for dv in result.intermediate_delta_v
        ],
        "total_time_days": result.total_time_of_flight.to_value(u.day),
    }

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    args = parse_args()
    optimize_sequence(
        sequence_id=args.sequence_id,
        population=args.population,
        generations=args.generations,
        departure_offset_bounds=tuple(args.departure_offset_bounds),
        output=args.output,
    )


if __name__ == "__main__":
    main()


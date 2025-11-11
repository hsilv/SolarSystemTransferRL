from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Body

from baseline_lambert import (
    LambertSummary,
    check_transfer_collisions,
    find_collision_free_transfer,
    solve_lambert_transfer,
)


@dataclass(frozen=True)
class MgaLegResult:
    """Container for a single leg of a multi-gravity-assist trajectory."""

    summary: LambertSummary
    index: int
    impulsive_delta_v: u.Quantity
    elapsed_time: u.Quantity


@dataclass(frozen=True)
class MgaResult:
    """Aggregated result for a multi-gravity-assist transfer."""

    sequence: Tuple[Body, ...]
    departure_date: Time
    time_of_flights: Tuple[u.Quantity, ...]
    legs: Tuple[MgaLegResult, ...]
    initial_delta_v: u.Quantity
    intermediate_delta_v: Tuple[u.Quantity, ...]
    arrival_delta_v: u.Quantity
    total_delta_v: u.Quantity
    total_time_of_flight: u.Quantity


def _as_time_quantity(value: u.Quantity | float | int) -> u.Quantity:
    if isinstance(value, u.Quantity):
        return value.to(u.day)
    return float(value) * u.day


def _vector_magnitude(quantity: u.Quantity) -> u.Quantity:
    vector = quantity.to(u.km / u.s).value
    return np.linalg.norm(vector) * (u.km / u.s)


def evaluate_mga(
    departure_date: Time,
    sequence: Sequence[Body],
    time_of_flights: Sequence[u.Quantity | float | int],
) -> MgaResult:
    if len(sequence) < 2:
        raise ValueError("`sequence` must contain at least two bodies.")
    if len(time_of_flights) != len(sequence) - 1:
        raise ValueError(
            "Length of `time_of_flights` must match number of trajectory legs "
            "(len(sequence) - 1)."
        )

    sequence_tuple = tuple(sequence)
    tof_quantities: Tuple[u.Quantity, ...] = tuple(_as_time_quantity(tof) for tof in time_of_flights)

    leg_summaries: List[MgaLegResult] = []
    intermediate_impulses: List[u.Quantity] = []

    current_departure_date = departure_date
    cumulative_time = 0 * u.day
    previous_summary: LambertSummary | None = None

    initial_delta_v = 0 * (u.km / u.s)
    arrival_delta_v = 0 * (u.km / u.s)

    for index, (departure_body, arrival_body, tof) in enumerate(
        zip(sequence_tuple[:-1], sequence_tuple[1:], tof_quantities)
    ):
        summary = solve_lambert_transfer(
            current_departure_date,
            tof,
            departure_body=departure_body,
            arrival_body=arrival_body,
        )

        if index == 0:
            initial_delta_v = summary.delta_v_departure
            impulsive_delta_v = initial_delta_v
        else:
            dv_impulse = _vector_magnitude(summary.v_departure - previous_summary.v_arrival)
            intermediate_impulses.append(dv_impulse)
            impulsive_delta_v = dv_impulse

        if index == len(sequence_tuple) - 2:
            arrival_delta_v = summary.delta_v_arrival

        cumulative_time += tof
        leg_summaries.append(
            MgaLegResult(
                summary=summary,
                index=index,
                impulsive_delta_v=impulsive_delta_v,
                elapsed_time=cumulative_time,
            )
        )

        current_departure_date = summary.arrival_date
        previous_summary = summary

    arrival_delta_v = arrival_delta_v.to(u.km / u.s)
    total_delta_v = initial_delta_v + arrival_delta_v
    for dv in intermediate_impulses:
        total_delta_v += dv

    return MgaResult(
        sequence=sequence_tuple,
        departure_date=departure_date,
        time_of_flights=tof_quantities,
        legs=tuple(leg_summaries),
        initial_delta_v=initial_delta_v,
        intermediate_delta_v=tuple(intermediate_impulses),
        arrival_delta_v=arrival_delta_v,
        total_delta_v=total_delta_v,
        total_time_of_flight=sum(tof_quantities, 0 * u.day),
    )


def evaluate_mga_safe(
    departure_date: Time,
    sequence: Sequence[Body],
    time_of_flights: Sequence[u.Quantity | float | int],
    collision_frames: int = 400,
    search_step_days: int = 15,
    departure_window_days: int = 120,
    tof_window_days: int = 120,
    enforce_collision_free: bool = True,
    verbose: bool = False,
) -> MgaResult:
    if len(sequence) < 2:
        raise ValueError("`sequence` must contain at least two bodies.")
    if len(time_of_flights) != len(sequence) - 1:
        raise ValueError(
            "Length of `time_of_flights` must match number of trajectory legs "
            "(len(sequence) - 1)."
        )

    sequence_tuple = tuple(sequence)
    base_tofs: Tuple[u.Quantity, ...] = tuple(_as_time_quantity(tof) for tof in time_of_flights)

    if not enforce_collision_free:
        return evaluate_mga(departure_date, sequence_tuple, base_tofs)

    leg_results: List[MgaLegResult] = []
    intermediate_impulses: List[u.Quantity] = []

    current_departure = departure_date
    previous_summary: LambertSummary | None = None

    initial_delta_v = 0 * (u.km / u.s)
    arrival_delta_v = 0 * (u.km / u.s)
    adjusted_tofs: List[u.Quantity] = []

    for index, tof in enumerate(base_tofs):
        departure_body = sequence_tuple[index]
        arrival_body = sequence_tuple[index + 1]

        if verbose:
            print(
                f"[MGA] Leg {index + 1}/{len(sequence_tuple) - 1}: "
                f"{departure_body.name} → {arrival_body.name}, "
                f"TOF base {tof.to(u.day).value:.1f} d"
            )
        summary = solve_lambert_transfer(
            current_departure,
            tof,
            departure_body=departure_body,
            arrival_body=arrival_body,
        )
        collision = check_transfer_collisions(summary, frames=collision_frames)
        if collision:
            if verbose:
                print(
                    f"[MGA] Colisión con {collision}. "
                    "Buscando alternativa sin colisiones..."
                )
            try:
                summary = find_collision_free_transfer(
                    base_departure=current_departure,
                    base_time_of_flight=tof,
                    frames=collision_frames,
                    departure_window_days=departure_window_days,
                    tof_window_days=tof_window_days,
                    step_days=search_step_days,
                    departure_body=departure_body,
                    arrival_body=arrival_body,
                    verbose=verbose,
                )
            except RuntimeError:
                if enforce_collision_free:
                    raise
                if verbose:
                    print("[MGA] No se encontró alternativa libre de colisión; se usará la trayectoria original.")

        adjusted_tof = summary.time_of_flight
        adjusted_tofs.append(adjusted_tof)

        if index == 0:
            initial_delta_v = summary.delta_v_departure
            impulsive_delta_v = initial_delta_v
        else:
            dv_impulse = _vector_magnitude(summary.v_departure - previous_summary.v_arrival)
            intermediate_impulses.append(dv_impulse)
            impulsive_delta_v = dv_impulse

        if index == len(sequence_tuple) - 2:
            arrival_delta_v = summary.delta_v_arrival

        cumulative_time = adjusted_tof if not leg_results else leg_results[-1].elapsed_time + adjusted_tof
        leg_results.append(
            MgaLegResult(
                summary=summary,
                index=index,
                impulsive_delta_v=impulsive_delta_v,
                elapsed_time=cumulative_time,
            )
        )

        current_departure = summary.arrival_date
        previous_summary = summary

    arrival_delta_v = arrival_delta_v.to(u.km / u.s)
    total_delta_v = initial_delta_v + arrival_delta_v
    for dv in intermediate_impulses:
        total_delta_v += dv

    total_time = sum(adjusted_tofs, 0 * u.day)

    return MgaResult(
        sequence=sequence_tuple,
        departure_date=departure_date,
        time_of_flights=tuple(adjusted_tofs),
        legs=tuple(leg_results),
        initial_delta_v=initial_delta_v,
        intermediate_delta_v=tuple(intermediate_impulses),
        arrival_delta_v=arrival_delta_v,
        total_delta_v=total_delta_v,
        total_time_of_flight=total_time,
    )


def print_mga_result(
    sequence_id: str,
    result: MgaResult,
    tof_inputs_days: Iterable[float] | None = None,
) -> None:
    tof_inputs = list(tof_inputs_days) if tof_inputs_days is not None else []
    print(f"Sequence {sequence_id}: {result.sequence[0].name} → {result.sequence[-1].name}")
    print(
        f"Departure date (TDB): {result.departure_date.tdb.iso} | Arrival date (TDB): "
        f"{result.legs[-1].summary.arrival_date.tdb.iso}"
    )
    print(f"Total Δv: {result.total_delta_v.to(u.km / u.s):.3f}")
    print(f"Total TOF: {result.total_time_of_flight.to(u.day).value:.1f} days")
    for idx, leg in enumerate(result.legs):
        actual_tof = leg.summary.time_of_flight.to(u.day).value
        tof_segment = f"TOF {actual_tof:.1f} d"
        if idx < len(tof_inputs):
            tof_segment += f" (input {tof_inputs[idx]:.1f} d)"
        print(
            f"  Leg {leg.index + 1}: {leg.summary.departure_body.name} → {leg.summary.arrival_body.name} | "
            f"{tof_segment} | Δv impulse {leg.impulsive_delta_v.to(u.km / u.s):.3f}"
        )
    if result.intermediate_delta_v:
        for idx, dv in enumerate(result.intermediate_delta_v, start=1):
            print(f"  Flyby impulse {idx}: {dv.to(u.km / u.s):.3f}")
    print(
        f"  Departure Δv: {result.initial_delta_v.to(u.km / u.s):.3f} | "
        f"Arrival Δv: {result.arrival_delta_v.to(u.km / u.s):.3f}"
    )


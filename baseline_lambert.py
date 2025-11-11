from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import (
    Earth,
    Jupiter,
    Mars,
    Mercury,
    Neptune,
    Saturn,
    Sun,
    Uranus,
    Venus,
)
from poliastro.iod import izzo
from poliastro.ephem import Ephem
from poliastro.twobody.orbit import Orbit

from refinement_data import record_refinement_sample, suggest_refinement_offsets

SOLAR_SYSTEM_BODIES: Iterable = (
    Mercury,
    Venus,
    Earth,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
)

COLLISION_SAFETY_FACTOR = 1.1
PLANET_COLORS: Dict[str, str] = {
    "Mercury": "#a6a6a6",
    "Venus": "#d4aa00",
    "Earth": "#2b7fff",
    "Mars": "#c1440e",
    "Jupiter": "#c7a26c",
    "Saturn": "#d8c573",
    "Uranus": "#66ccff",
    "Neptune": "#4063d8",
}

@dataclass
class LambertSummary:
    """Container with the main results of an interplanetary Lambert transfer."""

    departure_body: Any
    arrival_body: Any
    departure_date: Time
    arrival_date: Time
    time_of_flight: u.Quantity
    v_departure: u.Quantity
    v_arrival: u.Quantity
    vinf_departure: u.Quantity
    vinf_arrival: u.Quantity
    delta_v_departure: u.Quantity
    delta_v_arrival: u.Quantity
    total_delta_v: u.Quantity
    transfer_orbit: Orbit

def _state_from_body(body, epoch: Time) -> Orbit:
    """Return the heliocentric orbit of `body` at `epoch`."""
    ephem = Ephem.from_body(body, epoch)
    r_vec, v_vec = ephem.rv(epoch)
    return Orbit.from_vectors(Sun, r_vec, v_vec, epoch=epoch)

def _vector_magnitude(quantity: u.Quantity) -> u.Quantity:
    """Return the magnitude of a 3D vector quantity expressed in km/s."""
    vector = quantity.to(u.km / u.s).value
    return np.linalg.norm(vector) * (u.km / u.s)

def _sample_orbit_xy(orbit: Orbit, points: int, unit: u.Unit) -> np.ndarray:
    samples = orbit.sample(points)
    return np.column_stack(
        (
            samples.x.to_value(unit),
            samples.y.to_value(unit),
        )
    )

def _body_positions(
    body,
    epochs,
    unit: u.Unit,
) -> np.ndarray:
    """Return body positions for each epoch expressed in `unit`."""
    return np.array(
        [_state_from_body(body, epoch).r.to(unit).value for epoch in epochs]
    )

def _collision_radii() -> Dict[str, u.Quantity]:
    """Return safety radii per planet (planetary radius times safety factor)."""
    radii = {}
    for body in SOLAR_SYSTEM_BODIES:
        radius = getattr(body, "R", None)
        if radius is None:
            continue
        radii[body.name] = COLLISION_SAFETY_FACTOR * radius
    return radii


def check_transfer_collisions(
    summary: LambertSummary, frames: int = 400
) -> Optional[str]:
    """Return the name of the first planet with a potential collision, or None."""
    unit = u.km
    time_offsets = np.linspace(0.0, 1.0, frames) * summary.time_of_flight
    transfer_positions = np.array(
        [
            summary.transfer_orbit.propagate(offset).r.to(unit).value
            for offset in time_offsets
        ]
    )
    time_grid = [summary.departure_date + offset for offset in time_offsets]

    collision_radii = _collision_radii()
    for body in SOLAR_SYSTEM_BODIES:
        radius = collision_radii.get(body.name)
        if radius is None:
            continue
        body_positions = _body_positions(body, time_grid, unit)
        distances = np.linalg.norm(transfer_positions - body_positions, axis=1)
        if np.any(distances < radius.to_value(unit)):
            return body.name
    return None

def solve_lambert_transfer(
    departure_date: Time,
    time_of_flight: u.Quantity,
    departure_body=Earth,
    arrival_body=Saturn,
) -> LambertSummary:
    """Solve a heliocentric Lambert transfer between two bodies."""
    departure_orbit = _state_from_body(departure_body, departure_date)
    arrival_date = departure_date + time_of_flight
    arrival_orbit = _state_from_body(arrival_body, arrival_date)

    r_departure = departure_orbit.r
    r_arrival = arrival_orbit.r

    v_departure, v_arrival = izzo.lambert(
        Sun.k,
        r_departure,
        r_arrival,
        time_of_flight,
    )

    vinf_departure = (v_departure - departure_orbit.v).to(u.km / u.s)
    vinf_arrival = (v_arrival - arrival_orbit.v).to(u.km / u.s)
    delta_v_departure = _vector_magnitude(vinf_departure)
    delta_v_arrival = _vector_magnitude(vinf_arrival)
    total_delta_v = delta_v_departure + delta_v_arrival

    transfer_orbit = Orbit.from_vectors(
        Sun,
        r_departure,
        v_departure,
        epoch=departure_date,
    )

    return LambertSummary(
        departure_body=departure_body,
        arrival_body=arrival_body,
        departure_date=departure_date,
        arrival_date=arrival_date,
        time_of_flight=time_of_flight,
        v_departure=v_departure,
        v_arrival=v_arrival,
        vinf_departure=vinf_departure,
        vinf_arrival=vinf_arrival,
        delta_v_departure=delta_v_departure,
        delta_v_arrival=delta_v_arrival,
        total_delta_v=total_delta_v,
        transfer_orbit=transfer_orbit,
    )

def _print_summary(summary: LambertSummary) -> None:
    print(f"Lambert transfer {summary.departure_body.name} -> {summary.arrival_body.name}")
    print("--------------------------------")
    print(f"Departure date (TDB): {summary.departure_date.tdb.iso}")
    print(f"Arrival date (TDB):   {summary.arrival_date.tdb.iso}")
    print(
        "Time of flight:       "
        f"{summary.time_of_flight.to(u.day).value:.1f} day "
        f"({summary.time_of_flight.to(u.year).value:.2f} year)"
    )
    print()
    print("Heliocentric velocities")
    print(f"  Departure: {summary.v_departure.to(u.km / u.s)}")
    print(f"  Arrival:   {summary.v_arrival.to(u.km / u.s)}")
    print()
    print("Hyperbolic excess vectors (km/s)")
    print(f"  Departure v_inf: {summary.vinf_departure.to(u.km / u.s)}")
    print(f"  Arrival v_inf:   {summary.vinf_arrival.to(u.km / u.s)}")
    print()
    print("Hyperbolic excess (relative to planets)")
    print(f"  Departure Δv: {summary.delta_v_departure.to(u.km / u.s)}")
    print(f"  Arrival Δv:   {summary.delta_v_arrival.to(u.km / u.s)}")
    print(f"  Total Δv:     {summary.total_delta_v.to(u.km / u.s)}")

def plot_lambert_transfer(
    summary: LambertSummary,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Render a 2D plot of the Lambert transfer with all planetary orbits."""
    unit = u.AU
    orbit_samples = {
        body.name: _sample_orbit_xy(_state_from_body(body, summary.departure_date), 720, unit)
        for body in SOLAR_SYSTEM_BODIES
    }
    transfer_samples = _sample_orbit_xy(summary.transfer_orbit, 720, unit)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    for body in SOLAR_SYSTEM_BODIES:
        coords = orbit_samples[body.name]
        color = PLANET_COLORS.get(body.name)
        linewidth = 1.4 if body in (Jupiter, Saturn) else 1.0
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            label=f"{body.name} orbit",
            color=color,
            linewidth=linewidth,
        )

    ax.plot(
        transfer_samples[:, 0],
        transfer_samples[:, 1],
        label="Lambert arc",
        color="black",
        linewidth=1.6,
    )

    all_x = np.concatenate([coords[:, 0] for coords in orbit_samples.values()] + [transfer_samples[:, 0]])
    all_y = np.concatenate([coords[:, 1] for coords in orbit_samples.values()] + [transfer_samples[:, 1]])

    x_span = all_x.max() - all_x.min()
    y_span = all_y.max() - all_y.min()
    span = max(x_span, y_span)
    margin = 0.1 * span
    center_x = 0.5 * (all_x.max() + all_x.min())
    center_y = 0.5 * (all_y.max() + all_y.min())

    ax.set_xlim(center_x - span / 2 - margin, center_x + span / 2 + margin)
    ax.set_ylim(center_y - span / 2 - margin, center_y + span / 2 + margin)
    ax.set_title("Lambert Transfer with Solar System Context")
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

def animate_lambert_transfer(
    summary: LambertSummary,
    frames: int = 200,
    interval: int = 40,
    output_path: Optional[str] = None,
    show: bool = True,
    allow_collisions: bool = False,
    bodies: Iterable = SOLAR_SYSTEM_BODIES,
) -> FuncAnimation:
    """Animate the Lambert transfer and optionally persist it as a GIF."""
    if frames < 2:
        raise ValueError("Animation requires at least two frames.")

    if not allow_collisions:
        collision_body = check_transfer_collisions(summary, frames=frames)
        if collision_body:
            raise RuntimeError(
                f"Potential collision detected with {collision_body}. "
                "Adjust departure date or time of flight."
            )

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    unit_plot = u.AU
    unit_collision = u.km

    orbit_samples_plot: Dict[str, np.ndarray] = {}
    bodies = tuple(bodies)

    for body in bodies:
        orbit = _state_from_body(body, summary.departure_date)
        orbit_samples_plot[body.name] = _sample_orbit_xy(orbit, 720, unit_plot)
        ax.plot(
            orbit_samples_plot[body.name][:, 0],
            orbit_samples_plot[body.name][:, 1],
            label=f"{body.name} orbit",
            color=PLANET_COLORS.get(body.name),
            linewidth=1.0 if body not in (Jupiter, Saturn) else 1.3,
        )

    time_grid = [
        summary.departure_date + frac * summary.time_of_flight
        for frac in np.linspace(0.0, 1.0, frames)
    ]
    time_offsets = np.linspace(0.0, 1.0, frames) * summary.time_of_flight

    transfer_positions_plot = np.array(
        [
            summary.transfer_orbit.propagate(offset).r.to(unit_plot).value
            for offset in time_offsets
        ]
    )
    transfer_positions_collision = np.array(
        [
            summary.transfer_orbit.propagate(offset).r.to(unit_collision).value
            for offset in time_offsets
        ]
    )
    ax.plot(
        transfer_positions_plot[:, 0],
        transfer_positions_plot[:, 1],
        label="Lambert arc",
        color="black",
        linewidth=1.6,
    )

    body_positions_plot: Dict[str, np.ndarray] = {}
    body_positions_collision: Dict[str, np.ndarray] = {}
    for body in bodies:
        body_positions_plot[body.name] = _body_positions(
            body,
            time_grid,
            unit_plot,
        )
        body_positions_collision[body.name] = _body_positions(
            body,
            time_grid,
            unit_collision,
        )

    collision_radii = _collision_radii()
    for body in SOLAR_SYSTEM_BODIES:
        body_name = body.name
        if body_name not in collision_radii:
            continue
        if not allow_collisions:
            distances = np.linalg.norm(
                transfer_positions_collision - body_positions_collision[body_name],
                axis=1,
            )
            if np.any(distances < collision_radii[body_name].to_value(unit_collision)):
                raise RuntimeError(
                    f"Potential collision detected with {body_name}. "
                    "Adjust departure date or time of flight."
                )

    markers: Dict[str, Any] = {}
    for body in bodies:
        color = PLANET_COLORS.get(body.name)
        marker, = ax.plot(
            [],
            [],
            "o",
            label="_nolegend_",
            color=color,
        )
        markers[body.name] = marker

    probe_marker, = ax.plot([], [], "o", color="C3", label="Spacecraft")
    probe_trail, = ax.plot([], [], color="C3", linestyle="--", alpha=0.7)

    orbit_bounds_x = [samples[:, 0] for samples in orbit_samples_plot.values()]
    orbit_bounds_y = [samples[:, 1] for samples in orbit_samples_plot.values()]
    all_x = np.concatenate(orbit_bounds_x + [transfer_positions_plot[:, 0]])
    all_y = np.concatenate(orbit_bounds_y + [transfer_positions_plot[:, 1]])

    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    margin = 0.15 * span
    center_x = 0.5 * (all_x.max() + all_x.min())
    center_y = 0.5 * (all_y.max() + all_y.min())
    ax.set_xlim(center_x - span / 2 - margin, center_x + span / 2 + margin)
    ax.set_ylim(center_y - span / 2 - margin, center_y + span / 2 + margin)
    ax.set_title("Lambert Transfer Across the Solar System")
    ax.set_xlabel(f"x [{unit_plot}]")
    ax.set_ylabel(f"y [{unit_plot}]")

    handles, labels = ax.get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label != "_nolegend_"
    ]
    combined = {}
    for handle, label in legend_items:
        combined[label] = handle
    ax.legend(combined.values(), combined.keys(), loc="upper right", ncol=2, fontsize=9)

    def init():
        for marker in markers.values():
            marker.set_data([], [])
        probe_marker.set_data([], [])
        probe_trail.set_data([], [])
        return (*markers.values(), probe_marker, probe_trail)

    def update(frame_index: int):
        for body_name, marker in markers.items():
            positions = body_positions_plot[body_name]
            marker.set_data(positions[frame_index, 0], positions[frame_index, 1])
        probe_marker.set_data(
            transfer_positions_plot[frame_index, 0],
            transfer_positions_plot[frame_index, 1],
        )
        probe_trail.set_data(
            transfer_positions_plot[: frame_index + 1, 0],
            transfer_positions_plot[: frame_index + 1, 1],
        )
        return (*markers.values(), probe_marker, probe_trail)

    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )

    if output_path:
        fps = max(1, int(round(1000 / interval)))
        writer = PillowWriter(fps=fps)
        animation.save(output_path, writer=writer)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


def find_collision_free_transfer(
    base_departure: Time,
    base_time_of_flight: u.Quantity,
    frames: int = 400,
    departure_window_days: int = 720,
    tof_window_days: int = 720,
    step_days: int = 30,
    departure_body=Earth,
    arrival_body=Saturn,
    verbose: bool = False,
) -> LambertSummary:
    """
    Search for a collision-free Lambert transfer near the provided baseline.

    Scans a grid of departure date offsets and time-of-flight offsets around the
    baseline values. Among collision-free solutions, returns the one with the
    lowest total delta-v. Raises RuntimeError if no safe transfer is found within
    the window.
    """
    date_offsets = range(-departure_window_days, departure_window_days + step_days, step_days)
    tof_offsets = range(-tof_window_days, tof_window_days + step_days, step_days)

    best_summary: Optional[LambertSummary] = None
    attempts = 0
    tested_offsets: Set[Tuple[int, int]] = set()

    suggestions = suggest_refinement_offsets(
        departure_body,
        arrival_body,
        date_offsets,
        tof_offsets,
    )

    def try_offset(date_offset: int, tof_offset: int) -> Optional[LambertSummary]:
        nonlocal attempts, best_summary
        if (date_offset, tof_offset) in tested_offsets:
            return None
        tested_offsets.add((date_offset, tof_offset))

        departure = base_departure + date_offset * u.day
        tof_candidate = base_time_of_flight + tof_offset * u.day
        if tof_candidate <= 0 * u.day:
            return None
        try:
            candidate = solve_lambert_transfer(
                departure,
                tof_candidate,
                departure_body=departure_body,
                arrival_body=arrival_body,
            )
        except Exception:
            return None

        attempts += 1
        collision = check_transfer_collisions(candidate, frames=frames)
        record_refinement_sample(
            departure_body,
            arrival_body,
            date_offset,
            tof_offset,
            candidate.total_delta_v.to_value(u.km / u.s),
            success=collision is None,
        )
        if verbose and attempts % 100 == 0:
            print(
                "[SEARCH] "
                f"{departure_body.name}->{arrival_body.name} intentos {attempts}, "
                f"Δv actual {candidate.total_delta_v.to_value(u.km / u.s):.3f} km/s"
            )
        if collision is not None:
            return None
        if best_summary is None or candidate.total_delta_v < best_summary.total_delta_v:
            best_summary = candidate
            if verbose:
                print(
                    "[SEARCH] "
                    f"Nuevo mejor Δv {best_summary.total_delta_v.to_value(u.km / u.s):.3f} km/s "
                    f"con salida {best_summary.departure_date.tdb.iso} "
                    f"y TOF {best_summary.time_of_flight.to(u.day).value:.1f} d"
                )
        return best_summary

    for suggestion in suggestions:
        result = try_offset(*suggestion)
        if result is not None:
            return result

    max_attempts = min(len(date_offsets) * len(tof_offsets), 400)
    for date_offset in date_offsets:
        for tof_offset in tof_offsets:
            result = try_offset(date_offset, tof_offset)
            if result is not None:
                return result
            if attempts >= max_attempts:
                break
        if attempts >= max_attempts:
            break

    if best_summary is None:
        import random

        remaining = [
            (d, t)
            for d in date_offsets
            for t in tof_offsets
            if (d, t) not in tested_offsets
        ]
        random.shuffle(remaining)
        for date_offset, tof_offset in remaining[:200]:
            result = try_offset(date_offset, tof_offset)
            if result is not None:
                return result
    if best_summary is None:
        raise RuntimeError(
            "Unable to find a collision-free transfer within the specified search window."
        )
    return best_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve a baseline Lambert transfer from Earth to Saturn."
    )
    parser.add_argument(
        "--departure",
        type=str,
        default="2030-01-01",
        help="Departure date in ISO format (default: 2030-01-01).",
    )
    parser.add_argument(
        "--tof-days",
        type=float,
        default=2555.0,
        help="Time of flight in days (default: 2555).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="route.png",
        help="Path to save the Lambert transfer image (default: route.png).",
    )
    args = parser.parse_args()

    departure_date = Time(args.departure, scale="tdb")
    if args.tof_days <= 0:
        parser.error("Time of flight must be a positive number of days.")
    time_of_flight = args.tof_days * u.day

    summary = solve_lambert_transfer(departure_date, time_of_flight)
    _print_summary(summary)
    if args.output:
        plot_lambert_transfer(summary, output_path=args.output, show=False)
        print(f"Route image saved to {args.output}")

if __name__ == "__main__":
    main()

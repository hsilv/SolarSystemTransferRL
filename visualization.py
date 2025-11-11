from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from baseline_lambert import (
    PLANET_COLORS,
    SOLAR_SYSTEM_BODIES,
    _sample_orbit_xy,
    _state_from_body,
    animate_lambert_transfer,
)
from mga import MgaResult


def plot_mga_route(
    result: MgaResult,
    output_path: str = "route.png",
    show: bool = True,
    sample_points: int = 512,
) -> None:
    """
    Plot a multi-leg Lambert trajectory together with planetary orbits.
    """
    unit = u.AU

    involved_names = {body.name for body in result.sequence}
    # Ensure departure and arrival bodies are included explicitly
    involved_names.add(result.sequence[0].name)
    involved_names.add(result.sequence[-1].name)

    orbit_samples = {
        body.name: _sample_orbit_xy(_state_from_body(body, result.departure_date), sample_points, unit)
        for body in SOLAR_SYSTEM_BODIES
        if body.name in involved_names
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")

    for body_name, coords in orbit_samples.items():
        color = PLANET_COLORS.get(body_name, "gray")
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            label=f"{body_name} orbit",
            color=color,
            linewidth=1.3 if body_name in ("Jupiter", "Saturn") else 1.0,
        )

    cmap = plt.get_cmap("plasma")
    leg_colors = _leg_colors(len(result.legs), cmap)
    transfer_samples = []
    for leg, color in zip(result.legs, leg_colors):
        transfer_coords = _sample_orbit_xy(
            leg.summary.transfer_orbit,
            sample_points,
            unit,
        )
        transfer_samples.append(transfer_coords)
        ax.plot(
            transfer_coords[:, 0],
            transfer_coords[:, 1],
            color=color,
            linewidth=1.8,
            label=f"Leg {leg.index + 1}: {leg.summary.departure_body.name} → {leg.summary.arrival_body.name}",
        )

        dep_orbit = _state_from_body(leg.summary.departure_body, leg.summary.departure_date)
        arr_orbit = _state_from_body(leg.summary.arrival_body, leg.summary.arrival_date)
        dep_coords = dep_orbit.r.to_value(unit)
        arr_coords = arr_orbit.r.to_value(unit)
        ax.scatter(dep_coords[0], dep_coords[1], color=color, marker="o", s=36)
        ax.scatter(arr_coords[0], arr_coords[1], color=color, marker="^", s=48)

    all_x = np.concatenate(
        [coords[:, 0] for coords in orbit_samples.values()]
        + [samples[:, 0] for samples in transfer_samples]
    )
    all_y = np.concatenate(
        [coords[:, 1] for coords in orbit_samples.values()]
        + [samples[:, 1] for samples in transfer_samples]
    )

    x_span = all_x.max() - all_x.min()
    y_span = all_y.max() - all_y.min()
    span = max(x_span, y_span)
    margin = 0.1 * span
    center_x = 0.5 * (all_x.max() + all_x.min())
    center_y = 0.5 * (all_y.max() + all_y.min())

    ax.set_xlim(center_x - span / 2 - margin, center_x + span / 2 + margin)
    ax.set_ylim(center_y - span / 2 - margin, center_y + span / 2 + margin)

    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.set_title("Multi-Gravity-Assist Lambert Trajectory")

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _leg_colors(count: int, cmap) -> Iterable:
    if count <= 1:
        yield cmap(0.6)
        return
    for idx in range(count):
        yield cmap(idx / max(1, count - 1))


def animate_mga_route(
    result: MgaResult,
    output_path: str = "mga_animation.gif",
    show: bool = True,
    frames_per_leg: int = 160,
    interval: int = 40,
    allow_collisions: bool = False,
) -> Sequence[str]:
    """
    Produce per-leg animations for a multi-gravity-assist trajectory.

    Returns a list of file paths for the generated animations.
    """
    outputs: list[str] = []
    for leg in result.legs:
        leg_output = output_path
        if len(result.legs) > 1:
            stem, suffix = output_path.rsplit(".", 1)
            leg_output = f"{stem}_leg{leg.index + 1}.{suffix}"
        print(
            f"[ANIMATE] Leg {leg.index + 1}/{len(result.legs)} "
            f"{leg.summary.departure_body.name} → {leg.summary.arrival_body.name}"
        )
        try:
            animate_lambert_transfer(
                leg.summary,
                frames=frames_per_leg,
                interval=interval,
                output_path=leg_output,
                show=show,
                allow_collisions=allow_collisions,
                bodies=(leg.summary.departure_body, leg.summary.arrival_body),
            )
            outputs.append(leg_output)
            print(f"[ANIMATE] Leg {leg.index + 1} guardado en {leg_output}")
        except RuntimeError as exc:
            print(
                f"[WARNING] Unable to animate leg {leg.index + 1} ({leg.summary.departure_body.name} → {leg.summary.arrival_body.name}): {exc}"
            )
    return outputs


from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import gymnasium as gym
import numpy as np
from astropy import units as u

from baseline_lambert import check_transfer_collisions
from mga import evaluate_mga
from sequences import SEQUENCE_CATALOG, SequenceDefinition


class MGAEnv(gym.Env):
    """Single-step environment to evaluate multi-gravity-assist transfers."""

    metadata: Dict[str, Any] = {}

    def __init__(
        self,
        reward_arrival: float = 1000.0,
        alpha: float = 25.0,
        beta: float = 0.2,
        departure_offset_bounds: Tuple[float, float] = (-365.0, 365.0),
        check_collisions: bool = False,
        collision_frames: int = 200,
        allowed_sequence_ids: Iterable[str] | None = None,
        flyby_bonus: float = 0.0,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("Penalty coefficients alpha and beta must be non-negative.")
        if allowed_sequence_ids is not None:
            allowed = {seq_id for seq_id in allowed_sequence_ids}
            filtered = tuple(seq for seq in SEQUENCE_CATALOG if seq.identifier in allowed)
            if not filtered:
                raise ValueError("allowed_sequence_ids does not match any known sequences.")
            self._catalog = filtered
        else:
            self._catalog = SEQUENCE_CATALOG
        self._reward_arrival = reward_arrival
        self._alpha = alpha
        self._beta = beta
        self._departure_offset_bounds = departure_offset_bounds
        self._check_collisions = check_collisions
        self._collision_frames = collision_frames
        self._flyby_bonus = flyby_bonus

        self._sequence_count = len(self._catalog)
        self._max_legs = max(len(seq.bodies) - 1 for seq in self._catalog)

        action_length = 2 + self._max_legs
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_length,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self._last_info: Dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._last_info = None
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}.")

        sequence_index = self._decode_sequence_index(action[0])
        sequence_def = self._catalog[sequence_index]

        departure_offset_days = self._scale_value(
            action[1],
            self._departure_offset_bounds,
        )
        tof_days = self._decode_time_of_flights(sequence_def, action[2:])

        departure_date = sequence_def.reference_departure + departure_offset_days * u.day

        terminated = True
        truncated = False
        reward = -1e6
        info: Dict[str, Any] = {
            "sequence_id": sequence_def.identifier,
            "departure_offset_days": float(departure_offset_days),
            "time_of_flights_days": tuple(float(v) for v in tof_days),
        }

        try:
            result = evaluate_mga(
                departure_date=departure_date,
                sequence=sequence_def.bodies,
                time_of_flights=tof_days,
            )
            collision_body = None
            if self._check_collisions:
                for leg in result.legs:
                    collision_body = check_transfer_collisions(
                        leg.summary,
                        frames=self._collision_frames,
                    )
                    if collision_body:
                        break

            total_delta_v = result.total_delta_v.to_value(u.km / u.s)
            total_time_days = result.total_time_of_flight.to_value(u.day)

            if collision_body:
                reward = -1e5
                info["collision_body"] = collision_body
            else:
                reward = (
                    self._reward_arrival
                    - self._alpha * total_delta_v
                    - self._beta * total_time_days
                    + self._flyby_bonus * max(0, len(sequence_def.bodies) - 2)
                )

            info.update(
                {
                    "delta_v_total_kms": total_delta_v,
                    "delta_v_initial_kms": result.initial_delta_v.to_value(u.km / u.s),
                    "delta_v_arrival_kms": result.arrival_delta_v.to_value(u.km / u.s),
                    "delta_v_intermediate_kms": tuple(
                        dv.to_value(u.km / u.s) for dv in result.intermediate_delta_v
                    ),
                    "total_time_days": total_time_days,
                    "arrival_date_tdb": result.legs[-1].summary.arrival_date.tdb.iso,
                    "departure_date_tdb": result.departure_date.tdb.iso,
                }
            )
            self._last_info = info
        except Exception as exc:
            reward = -1e6
            info["error"] = str(exc)
            self._last_info = info

        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, reward, terminated, truncated, info

    def render(self):
        return self._last_info

    def _decode_sequence_index(self, value: float) -> int:
        scaled = (value + 1.0) * 0.5 * (self._sequence_count - 1)
        index = int(round(np.clip(scaled, 0.0, self._sequence_count - 1)))
        return index

    def _decode_time_of_flights(
        self,
        sequence_def: SequenceDefinition,
        action_slice: np.ndarray,
    ) -> Tuple[float, ...]:
        tof_values: list[float] = []
        for leg_index, bounds in enumerate(sequence_def.tof_bounds_days):
            raw_value = action_slice[leg_index] if leg_index < len(action_slice) else 0.0
            tof_values.append(self._scale_value(raw_value, bounds))
        return tuple(tof_values)

    @staticmethod
    def _scale_value(value: float, bounds: Tuple[float, float]) -> float:
        clipped = float(np.clip(value, -1.0, 1.0))
        lower, upper = bounds
        half_range = 0.5 * (upper - lower)
        midpoint = lower + half_range
        return midpoint + clipped * half_range


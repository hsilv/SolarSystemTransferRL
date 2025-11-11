from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from rl_env import MGAEnv


class BestTrajectoryCallback(BaseCallback):
    """Track the best-performing trajectory during training."""

    def __init__(self, output_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self._output_path = output_path
        self._best_reward = float("-inf")
        self._best_info: Optional[Dict[str, Any]] = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        for info, reward in zip(infos, rewards):
            if info is None:
                continue
            if "error" in info or info.get("collision_body"):
                continue
            reward_value = float(reward)
            if reward_value > self._best_reward:
                self._best_reward = reward_value
                stored = info.copy()
                stored["reward"] = reward_value
                self._best_info = stored
        return True

    def _on_training_end(self) -> None:
        if self._best_info is None:
            return
        serializable = _to_serializable(self._best_info)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(json.dumps(serializable, indent=2))
        self._best_info = serializable

    @property
    def best_info(self) -> Optional[Dict[str, Any]]:
        return self._best_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for MGA planning.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Number of training timesteps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("ppo_mga.zip"),
        help="Path to save the trained PPO model.",
    )
    parser.add_argument(
        "--output-trajectory",
        type=Path,
        default=Path("best_traj.json"),
        help="Path to save the best trajectory summary.",
    )
    parser.add_argument("--reward-arrival", type=float, default=1000.0, help="Reward for successful arrival.")
    parser.add_argument("--alpha", type=float, default=25.0, help="Δv penalty coefficient.")
    parser.add_argument("--beta", type=float, default=0.2, help="Time-of-flight penalty coefficient.")
    parser.add_argument(
        "--check-collisions",
        action="store_true",
        help="Enable collision checking for every leg (slower).",
    )
    parser.add_argument(
        "--allowed-sequences",
        type=str,
        nargs="+",
        help="Limit training to specific sequence identifiers.",
    )
    parser.add_argument(
        "--flyby-bonus",
        type=float,
        default=0.0,
        help="Additional reward per flyby leg (len(sequence) - 2).",
    )
    parser.add_argument(
        "--ppo-verbose",
        type=int,
        default=1,
        help="Verbosity level passed to the Stable-Baselines3 PPO constructor.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Display Stable-Baselines3 progress bar during training (requires SB3 ≥ 1.8).",
    )
    return parser.parse_args()


def run_training(
    timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    n_envs: int = 4,
    output_model: Path | str = Path("ppo_mga.zip"),
    output_trajectory: Path | str = Path("best_traj.json"),
    reward_arrival: float = 1000.0,
    alpha: float = 25.0,
    beta: float = 0.2,
    check_collisions: bool = False,
    allowed_sequences: Iterable[str] | None = None,
    flyby_bonus: float = 0.0,
    ppo_verbose: int = 1,
    progress_bar: bool = False,
) -> Dict[str, Any]:
    output_model_path = Path(output_model)
    output_trajectory_path = Path(output_trajectory)

    def factory():
        return MGAEnv(
            reward_arrival=reward_arrival,
            alpha=alpha,
            beta=beta,
            check_collisions=check_collisions,
            allowed_sequence_ids=allowed_sequences,
            flyby_bonus=flyby_bonus,
        )

    env = make_vec_env(factory, n_envs=n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        verbose=ppo_verbose,
    )

    callback = BestTrajectoryCallback(output_path=output_trajectory_path)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=progress_bar)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)

    result: Dict[str, Any] = {
        "model_path": str(output_model_path),
        "trajectory_path": str(output_trajectory_path),
        "best_info": callback.best_info,
    }
    return result


def main() -> None:
    args = parse_args()
    run_training(
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        n_envs=args.n_envs,
        output_model=args.output_model,
        output_trajectory=args.output_trajectory,
        reward_arrival=args.reward_arrival,
        alpha=args.alpha,
        beta=args.beta,
        check_collisions=args.check_collisions,
        allowed_sequences=args.allowed_sequences,
        flyby_bonus=args.flyby_bonus,
        ppo_verbose=args.ppo_verbose,
        progress_bar=args.progress_bar,
    )


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    main()


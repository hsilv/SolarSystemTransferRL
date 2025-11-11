from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from astropy import units as u
from astropy.time import Time

from baseline_lambert import (
    _print_summary,
    animate_lambert_transfer,
    check_transfer_collisions,
    find_collision_free_transfer,
    plot_lambert_transfer,
    solve_lambert_transfer,
)
from mga import evaluate_mga_safe, print_mga_result
from sequences import SEQUENCE_CATALOG, SequenceDefinition, get_sequence
from train_ppo import run_training
from visualization import plot_mga_route, animate_mga_route
from pygmo_benchmark import optimize_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Toolbox for baseline Lambert transfers, MGA evaluation, RL training, and benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command")

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run the baseline Earth → Saturn Lambert transfer (Stage 1).",
    )
    baseline_parser.add_argument("--departure", type=str, default="2031-07-14", help="Departure date in ISO format (TDB scale).")
    baseline_parser.add_argument("--tof-days", type=float, default=2555.0, help="Time of flight in days.")
    baseline_parser.add_argument("--output", type=str, default="route.png", help="Path to save the route plot.")
    baseline_parser.add_argument("--animate", action="store_true", help="Persist an animation of the transfer.")
    baseline_parser.add_argument("--animation-output", type=str, default="lambert_transfer.gif", help="Path to save the animation.")
    baseline_parser.add_argument("--search-safe", action="store_true", help="Search nearby solutions if a collision is detected.")
    baseline_parser.add_argument("--no-show", action="store_true", help="Skip displaying figures interactively.")
    baseline_parser.add_argument(
        "--animate-allow-collision",
        action="store_true",
        help="Generate the Lambert animation even when a potential collision is detected.",
    )

    mga_parser = subparsers.add_parser(
        "mga",
        help="Evaluate a multi-gravity-assist sequence (Stage 2/3).",
    )
    mga_parser.add_argument("sequence_id", type=str, help="Identifier of the sequence to evaluate.")
    mga_parser.add_argument("--departure-offset", type=float, default=0.0, help="Offset in days from the reference departure date.")
    mga_parser.add_argument(
        "--tof-days",
        type=float,
        nargs="*",
        help="Time of flight per leg in days. Omit to use the midpoint of catalog bounds.",
    )
    mga_parser.add_argument("--plot", action="store_true", help="Generate a multi-leg route plot.")
    mga_parser.add_argument("--output", type=str, default="route.png", help="Path to save the MGA plot when --plot is set.")
    mga_parser.add_argument("--no-show", action="store_true", help="Skip displaying figures interactively.")
    mga_parser.add_argument(
        "--skip-collision-refinement",
        action="store_true",
        help="Skip the local search that looks for collision-free alternatives.",
    )
    mga_parser.add_argument(
        "--animate-allow-collision",
        action="store_true",
        help="Allow animations to be generated even when collisions are detected.",
    )
    mga_parser.add_argument("--verbose", action="store_true", help="Print extra logs during MGA evaluation.")

    list_parser = subparsers.add_parser(
        "list-sequences",
        help="List available MGA sequence definitions (Stage 3).",
    )
    list_parser.add_argument("--verbose", action="store_true", help="Show detailed leg bounds.")

    rl_parser = subparsers.add_parser(
        "train",
        help="Train PPO agent for MGA planning (Stage 5).",
    )
    rl_parser.add_argument("--timesteps", type=int, default=200_000, help="Number of training timesteps.")
    rl_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    rl_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments.")
    rl_parser.add_argument("--output-model", type=str, default="ppo_mga.zip", help="Path to save the trained PPO model.")
    rl_parser.add_argument("--output-trajectory", type=str, default="best_traj.json", help="Path to save the best trajectory summary.")
    rl_parser.add_argument("--reward-arrival", type=float, default=1000.0, help="Reward for successful arrival.")
    rl_parser.add_argument("--alpha", type=float, default=25.0, help="Δv penalty coefficient.")
    rl_parser.add_argument("--beta", type=float, default=0.2, help="Time-of-flight penalty coefficient.")
    rl_parser.add_argument("--check-collisions", action="store_true", help="Enable collision checking during training (slower).")
    rl_parser.add_argument("--ppo-verbose", type=int, default=1, help="Verbosity level for the Stable-Baselines3 PPO model.")
    rl_parser.add_argument("--progress-bar", action="store_true", help="Show Stable-Baselines3 training progress bar (requires SB3 ≥ 1.8).")
    rl_parser.add_argument(
        "--allowed-sequences",
        type=str,
        nargs="+",
        help="Limit training to the provided sequence identifiers (e.g., E-J-S E-V-J-S).",
    )
    rl_parser.add_argument(
        "--flyby-bonus",
        type=float,
        default=0.0,
        help="Additional reward per flyby leg (len(sequence) - 2).",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run PyGMO global optimization for comparison (Stage 7).",
    )
    benchmark_parser.add_argument("--sequence-id", type=str, default="E-J-S", help="Sequence identifier to optimize.")
    benchmark_parser.add_argument("--population", type=int, default=64, help="Population size for differential evolution.")
    benchmark_parser.add_argument("--generations", type=int, default=200, help="Number of generations.")
    benchmark_parser.add_argument(
        "--departure-offset-bounds",
        type=float,
        nargs=2,
        default=(-365.0, 365.0),
        metavar=("MIN", "MAX"),
        help="Bounds for departure-date offset in days.",
    )
    benchmark_parser.add_argument("--output", type=str, default="pygmo_best.json", help="Path to save the optimizer result.")

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run baseline plotting, PPO training, MGA visualization, and optional benchmarking.",
    )
    pipeline_parser.add_argument("--baseline-departure", type=str, default="2031-07-14", help="Baseline departure date (ISO, TDB).")
    pipeline_parser.add_argument("--baseline-tof-days", type=float, default=2555.0, help="Baseline time of flight in days.")
    pipeline_parser.add_argument("--timesteps", type=int, default=50_000, help="Training timesteps for PPO.")
    pipeline_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for PPO.")
    pipeline_parser.add_argument("--n-envs", type=int, default=2, help="Number of parallel environments for PPO.")
    pipeline_parser.add_argument("--reward-arrival", type=float, default=1000.0, help="Reward for successful arrival in PPO.")
    pipeline_parser.add_argument("--alpha", type=float, default=25.0, help="Δv penalty coefficient.")
    pipeline_parser.add_argument("--beta", type=float, default=0.2, help="Time-of-flight penalty coefficient.")
    pipeline_parser.add_argument("--train-check-collisions", action="store_true", help="Enable collision checks during PPO training.")
    pipeline_parser.add_argument("--ppo-verbose", type=int, default=1, help="Verbosity level for the PPO model during pipeline training.")
    pipeline_parser.add_argument("--progress-bar", action="store_true", help="Show Stable-Baselines3 training progress bar during pipeline training.")
    pipeline_parser.add_argument("--skip-training", action="store_true", help="Skip PPO training stage.")
    pipeline_parser.add_argument("--force-retrain", action="store_true", help="Retrain PPO even if existing artifacts are present.")
    pipeline_parser.add_argument("--skip-benchmark", action="store_true", help="Skip PyGMO benchmarking stage.")
    pipeline_parser.add_argument("--sequence-id", type=str, default="E-J-S", help="Default sequence to evaluate if PPO produces no result.")
    pipeline_parser.add_argument("--animate-mga", action="store_true", help="Render animations for each MGA leg using the best trajectory.")
    pipeline_parser.add_argument(
        "--animate-mga-allow-collision",
        action="store_true",
        help="Render MGA leg animations even if a potential collision is detected.",
    )
    pipeline_parser.add_argument(
        "--skip-collision-refinement",
        action="store_true",
        help="Skip the local search that adjusts the MGA plan to avoid collisions.",
    )
    pipeline_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs during MGA evaluation and search.",
    )
    pipeline_parser.add_argument(
        "--allowed-sequences",
        type=str,
        nargs="+",
        help="Limit PPO training to the provided sequence identifiers.",
    )
    pipeline_parser.add_argument(
        "--flyby-bonus",
        type=float,
        default=50.0,
        help="Additional reward per flyby leg during PPO training.",
    )
    pipeline_parser.add_argument(
        "--allow-direct-sequence",
        action="store_true",
        help="Include the direct Earth→Saturn sequence when training PPO (default: exclude).",
    )
    pipeline_parser.add_argument("--population", type=int, default=32, help="PyGMO population size when benchmark runs.")
    pipeline_parser.add_argument("--generations", type=int, default=150, help="PyGMO generations when benchmark runs.")
    pipeline_parser.add_argument(
        "--departure-offset-bounds",
        type=float,
        nargs=2,
        default=(-365.0, 365.0),
        metavar=("MIN", "MAX"),
        help="PyGMO bounds for departure-date offset in days when benchmark runs.",
    )
    pipeline_parser.add_argument("--output-dir", type=str, default="pipeline_outputs", help="Directory to store generated artifacts.")

    return parser


def run_baseline(args: argparse.Namespace) -> None:
    departure_date = Time(args.departure, scale="tdb")
    if args.tof_days <= 0:
        raise ValueError("Time of flight must be positive.")
    time_of_flight = args.tof_days * u.day

    summary = solve_lambert_transfer(departure_date, time_of_flight)
    collision_body = check_transfer_collisions(summary)
    if collision_body and args.search_safe:
        try:
            summary = find_collision_free_transfer(
                base_departure=departure_date,
                base_time_of_flight=time_of_flight,
            )
            print(
                f"[INFO] Using collision-free transfer with departure {summary.departure_date.tdb.iso} "
                f"and TOF {summary.time_of_flight.to(u.day).value:.1f} days."
            )
            collision_body = None
        except RuntimeError as exc:
            print(f"[WARNING] {exc}")
    elif collision_body:
        print(f"[WARNING] Potential collision detected with {collision_body}.")

    _print_summary(summary)
    if args.output:
        plot_lambert_transfer(summary, output_path=args.output, show=not args.no_show)
        print(f"[INFO] Lambert route saved to {args.output}")
    if args.animate:
        try:
            animate_lambert_transfer(
                summary,
                frames=240,
                interval=40,
                output_path=args.animation_output,
                show=not args.no_show,
                allow_collisions=args.animate_allow_collision,
            )
            print(f"[INFO] Animation saved to {args.animation_output}")
        except RuntimeError as exc:
            print(f"[WARNING] {exc}")


def run_mga(args: argparse.Namespace) -> None:
    sequence = get_sequence(args.sequence_id)
    departure_date = sequence.reference_departure + args.departure_offset * u.day
    tof_days = resolve_tofs(sequence, args.tof_days)

    if args.skip_collision_refinement:
        print("[INFO] Skipping collision refinement; using provided TOFs as-is.")
    try:
        result = evaluate_mga_safe(
            departure_date=departure_date,
            sequence=sequence.bodies,
            time_of_flights=tof_days,
            enforce_collision_free=not args.skip_collision_refinement,
            verbose=args.verbose,
        )
    except RuntimeError as exc:
        print(f"[WARNING] {exc}")
        print("[INFO] Falling back to PyGMO optimization for a feasible trajectory.")
        pygmo_summary = optimize_sequence(
            sequence_id=sequence.identifier,
            departure_offset_bounds=(-180.0, 180.0),
        )
        departure_date = sequence.reference_departure + pygmo_summary["departure_offset_days"] * u.day
        tof_days = pygmo_summary["time_of_flights_days"]
        result = evaluate_mga_safe(
            departure_date=departure_date,
            sequence=sequence.bodies,
            time_of_flights=tof_days,
            enforce_collision_free=False,
            verbose=args.verbose,
        )
    print_mga_result(sequence.identifier, result, tof_days)

    if args.plot:
        plot_mga_route(result, output_path=args.output, show=not args.no_show)
        print(f"[INFO] MGA route saved to {args.output}")
        if args.animate_allow_collision:
            animation_path = args.output.rsplit(".", 1)[0] + ".gif"
            animate_mga_route(
                result,
                output_path=animation_path,
                show=not args.no_show,
                allow_collisions=True,
            )
            print(f"[INFO] MGA animation saved to {animation_path}")


def resolve_tofs(sequence: SequenceDefinition, override: Iterable[float] | None) -> List[float]:
    if override is None:
        return [0.5 * (low + high) for low, high in sequence.tof_bounds_days]
    override_list = list(override)
    if not override_list:
        return [0.5 * (low + high) for low, high in sequence.tof_bounds_days]
    if len(override_list) != len(sequence.tof_bounds_days):
        raise ValueError(
            "Number of provided TOF values must match the number of legs in the sequence."
        )
    return override_list


def print_mga_result(sequence_id: str, result: MgaResult, tof_days: Iterable[float]) -> None:
    tof_iter = list(tof_days)
    print(f"Sequence {sequence_id}: {result.legs[0].summary.departure_body.name} → {result.legs[-1].summary.arrival_body.name}")
    print(
        f"Departure date (TDB): {result.departure_date.tdb.iso} | Arrival date (TDB): "
        f"{result.legs[-1].summary.arrival_date.tdb.iso}"
    )
    print(f"Total Δv: {result.total_delta_v.to(u.km / u.s):.3f}")
    print(f"Total TOF: {result.total_time_of_flight.to(u.day).value:.1f} days")
    for leg, tof in zip(result.legs, tof_iter):
        print(
            f"  Leg {leg.index + 1}: {leg.summary.departure_body.name} → {leg.summary.arrival_body.name} | "
            f"TOF input {tof:.1f} d | Δv impulse {leg.impulsive_delta_v.to(u.km / u.s):.3f}"
        )
    if result.intermediate_delta_v:
        for idx, dv in enumerate(result.intermediate_delta_v, start=1):
            print(f"  Flyby impulse {idx}: {dv.to(u.km / u.s):.3f}")
    print(
        f"  Departure Δv: {result.initial_delta_v.to(u.km / u.s):.3f} | "
        f"Arrival Δv: {result.arrival_delta_v.to(u.km / u.s):.3f}"
    )


def list_sequences(verbose: bool) -> None:
    for sequence in SEQUENCE_CATALOG:
        description = " → ".join(body.name for body in sequence.bodies)
        print(f"{sequence.identifier}: {description}")
        if verbose:
            for leg_idx, bounds in enumerate(sequence.tof_bounds_days, start=1):
                print(
                    f"  Leg {leg_idx} bounds: {bounds[0]:.1f} d ≤ TOF ≤ {bounds[1]:.1f} d"
                )
            print(f"  Reference departure: {sequence.reference_departure.tdb.iso}")


def run_training_cli(args: argparse.Namespace) -> None:
    result = run_training(
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
    print(f"[INFO] PPO model saved to {result['model_path']}")
    print(f"[INFO] Best trajectory summary stored at {result['trajectory_path']}")
    best_info = result.get("best_info")
    if best_info and "reward" in best_info:
        print(f"[INFO] Best reward: {best_info['reward']:.3f}")


def run_benchmark(args: argparse.Namespace) -> None:
    summary = optimize_sequence(
        sequence_id=args.sequence_id,
        population=args.population,
        generations=args.generations,
        departure_offset_bounds=tuple(args.departure_offset_bounds),
        output=args.output,
    )
    print(f"[INFO] PyGMO summary saved to {args.output}")
    print(
        f"Best Δv: {summary['delta_v_total_kms']:.3f} km/s | Total TOF: {summary['total_time_days']:.1f} d"
    )


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.allowed_sequences:
        allowed_sequences = args.allowed_sequences
    elif args.allow_direct_sequence:
        allowed_sequences = None
    else:
        allowed_sequences = [seq.identifier for seq in SEQUENCE_CATALOG if seq.identifier != "E-S"]

    print("[STAGE 1] Solving baseline Lambert transfer...")
    baseline_departure = Time(args.baseline_departure, scale="tdb")
    if args.baseline_tof_days <= 0:
        raise ValueError("Baseline time of flight must be positive.")
    baseline_tof = args.baseline_tof_days * u.day
    baseline_summary = solve_lambert_transfer(baseline_departure, baseline_tof)
    collision_body = check_transfer_collisions(baseline_summary)
    if collision_body:
        print(f"[WARNING] Potential collision detected with {collision_body} during baseline transfer.")
    _print_summary(baseline_summary)
    baseline_plot = output_dir / "baseline_route.png"
    plot_lambert_transfer(baseline_summary, output_path=str(baseline_plot), show=False)
    print(f"[INFO] Baseline route stored at {baseline_plot}")

    best_info = None
    model_path = output_dir / "ppo_mga.zip"
    trajectory_path = output_dir / "best_traj.json"
    artifacts_exist = model_path.exists() and trajectory_path.exists()

    training_required = True
    if args.skip_training:
        print("[STAGE 5] Skipping PPO training as requested.")
        if artifacts_exist:
            candidate = json.loads(trajectory_path.read_text())
            if allowed_sequences and candidate.get("sequence_id") not in allowed_sequences:
                print("[WARNING] Stored PPO trajectory usa una secuencia no permitida. Se requiere reentrenar.")
            else:
                best_info = candidate
                print(f"[INFO] Reusing existing PPO artifacts from {output_dir}.")
                training_required = False
        else:
            print("[WARNING] No existing PPO artifacts found; PPO training se realizará igualmente.")
    elif artifacts_exist and not args.force_retrain:
        candidate = json.loads(trajectory_path.read_text())
        if allowed_sequences and candidate.get("sequence_id") not in allowed_sequences:
            print("[WARNING] Stored PPO trajectory usa una secuencia no permitida. Reentrenando.")
        else:
            print("[STAGE 5] Existing PPO artifacts detected; reusing without retraining.")
            best_info = candidate
            training_required = False

    if training_required:
        print("[STAGE 5] Training PPO agent...")
        training_result = run_training(
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            n_envs=args.n_envs,
            output_model=model_path,
            output_trajectory=trajectory_path,
            reward_arrival=args.reward_arrival,
            alpha=args.alpha,
            beta=args.beta,
            check_collisions=args.train_check_collisions,
            allowed_sequences=allowed_sequences,
            flyby_bonus=args.flyby_bonus,
            ppo_verbose=args.ppo_verbose,
            progress_bar=args.progress_bar,
        )
        print(f"[INFO] PPO model saved to {training_result['model_path']}")
        print(f"[INFO] Best trajectory JSON stored at {training_result['trajectory_path']}")
        best_info = training_result.get("best_info")
        if best_info and "reward" in best_info:
            print(f"[INFO] Best trajectory reward: {best_info['reward']:.3f}")
        elif not best_info:
            print("[WARNING] PPO training did not yield a valid trajectory. Using default sequence parameters.")

    print("[STAGE 6] Generating MGA trajectory plot...")
    sequence_id = args.sequence_id
    departure_offset_days = 0.0
    tof_days: List[float] | None = None
    if best_info:
        sequence_id = best_info.get("sequence_id", sequence_id)
        departure_offset_days = float(best_info.get("departure_offset_days", 0.0))
        tof_days_raw = best_info.get("time_of_flights_days")
        if tof_days_raw:
            tof_days = [float(value) for value in tof_days_raw]
    sequence = get_sequence(sequence_id)
    if tof_days is None:
        tof_days = [0.5 * (low + high) for low, high in sequence.tof_bounds_days]
    departure_date = sequence.reference_departure + departure_offset_days * u.day
    if args.skip_collision_refinement:
        print("[STAGE 6] Skipping collision refinement; using PPO outputs directly.")
    try:
        mga_result = evaluate_mga_safe(
            departure_date=departure_date,
            sequence=sequence.bodies,
            time_of_flights=tof_days,
            enforce_collision_free=not args.skip_collision_refinement,
            verbose=args.verbose,
        )
    except RuntimeError as exc:
        print(f"[WARNING] {exc}")
        print("[STAGE 6] Falling back to PyGMO optimization for a feasible trajectory.")
        pygmo_summary = optimize_sequence(
            sequence_id=sequence.identifier,
            departure_offset_bounds=(-180.0, 180.0),
        )
        departure_date = sequence.reference_departure + pygmo_summary["departure_offset_days"] * u.day
        tof_days = pygmo_summary["time_of_flights_days"]
        mga_result = evaluate_mga_safe(
            departure_date=departure_date,
            sequence=sequence.bodies,
            time_of_flights=tof_days,
            enforce_collision_free=False,
            verbose=args.verbose,
        )
    print_mga_result(sequence.identifier, mga_result, tof_days)
    mga_plot = output_dir / f"mga_{sequence_id}.png"
    plot_mga_route(mga_result, output_path=str(mga_plot), show=False)
    print(f"[INFO] MGA route stored at {mga_plot}")

    mga_animations: Iterable[str] | None = None
    if args.animate_mga:
        print("[STAGE 6] Rendering MGA leg animations...")
        mga_animation_path = output_dir / "mga_animation.gif"
        mga_animations = animate_mga_route(
            mga_result,
            output_path=str(mga_animation_path),
            show=False,
            allow_collisions=args.animate_mga_allow_collision,
        )
        for path in mga_animations:
            print(f"[INFO] MGA animation stored at {path}")

    if args.skip_benchmark:
        print("[STAGE 7] Skipping PyGMO benchmark as requested.")
        benchmark_summary = None
    else:
        print("[STAGE 7] Running PyGMO benchmark...")
        benchmark_summary = optimize_sequence(
            sequence_id=sequence_id,
            population=args.population,
            generations=args.generations,
            departure_offset_bounds=tuple(args.departure_offset_bounds),
            output=output_dir / "pygmo_best.json",
        )
        print(f"[INFO] PyGMO summary stored at {output_dir / 'pygmo_best.json'}")
        print(
            f"[INFO] PyGMO best Δv: {benchmark_summary['delta_v_total_kms']:.3f} km/s | "
            f"Total TOF: {benchmark_summary['total_time_days']:.1f} d"
        )

    print("[SUMMARY]")
    print(f"  Baseline route: {baseline_plot}")
    if args.skip_training:
        if artifacts_exist:
            print(f"  PPO model (reused): {model_path}")
            print(f"  PPO best trajectory (reused): {trajectory_path}")
    elif artifacts_exist and not args.force_retrain:
        print(f"  PPO model (reused): {model_path}")
        print(f"  PPO best trajectory (reused): {trajectory_path}")
    else:
        print(f"  PPO model: {model_path}")
        print(f"  PPO best trajectory: {trajectory_path}")
    print(f"  MGA route: {mga_plot}")
    if mga_animations:
        for path in mga_animations:
            print(f"  MGA animation: {path}")
    if not args.skip_benchmark and benchmark_summary:
        print(f"  PyGMO result: {output_dir / 'pygmo_best.json'}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(args)
    elif args.command == "mga":
        run_mga(args)
    elif args.command == "list-sequences":
        list_sequences(verbose=args.verbose)
    elif args.command == "train":
        run_training_cli(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

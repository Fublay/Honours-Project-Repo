"""Main application entry point for laser PID tuning.

This script keeps the user-facing flow together while the heavier tuning logic
now lives in focused modules under ``tuning/``.
"""

from __future__ import annotations

import csv
from datetime import datetime
import math
import random

import numpy as np
import serial

try:
    from skopt import gp_minimize

    HAVE_SKOPT = True
except Exception:
    gp_minimize = None
    HAVE_SKOPT = False

import laser_command_ids as CMD
from protocol.reply_parser import parse_ack
from transport.serial_interface import SerialLineIO
from tuning.bootstrap import (
    AXIS_NAMES,
    assess_bootstrap_progress,
    assess_local_safe_region,
    candidate_is_good,
    candidate_is_safe,
    choose_bootstrap_axis,
)
from tuning.metrics import compute_trial_metrics, score_controller
from tuning.search import (
    OnlineSurrogateModel,
    build_bayes_search_space,
    build_local_refinement_bounds,
    build_surrogate_training_row,
    clamp_pid_to_bounds,
    compute_refinement_step_sizes,
    filter_seed_points_for_space,
    propose_coordinate_candidate,
    propose_surrogate_candidate,
)
from tuning.trial_runner import run_trial
from ui.graphing import RuntimeMonitor, prompt_launch_gui, run_graph_tool


class EarlyStopOptimization(RuntimeError):
    pass


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ordered_row_fieldnames(rows: list[dict], fallback: list[str]) -> list[str]:
    """Preserve first-seen key order while including keys from every row."""
    if not rows:
        return list(fallback)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def prompt_launch_action() -> str:
    while True:
        choice = input("Choose action: [s]tart test, [r]eset defaults, [g]raph power, or [q]uit: ").strip().lower()
        if choice in {"s", "start"}:
            return "start"
        if choice in {"r", "reset", "defaults", "reset defaults"}:
            return "reset"
        if choice in {"g", "graph", "plot"}:
            return "graph"
        if choice in {"q", "quit", "exit"}:
            return "quit"
        print("Please enter s, r, g, or q.", flush=True)


def prompt_goal_power_output(default_value: float) -> float:
    while True:
        raw = input(f"Enter goal power output [{default_value}]: ").strip()
        if raw == "":
            return float(default_value)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a numeric value.", flush=True)


def prompt_trial_count(default_value: int) -> int:
    while True:
        raw = input(f"Enter number of trials [{default_value}]: ").strip()
        if raw == "":
            return int(default_value)
        try:
            value = int(raw)
            if value < 1:
                print("Please enter an integer >= 1.", flush=True)
                continue
            return value
        except ValueError:
            print("Please enter an integer value.", flush=True)


def prompt_frequency_khz(default_value: int) -> int:
    while True:
        raw = input(f"Enter frequency in kHz [{default_value}]: ").strip()
        if raw == "":
            return int(default_value)
        try:
            value = int(raw)
            if value < 0:
                print("Please enter an integer >= 0.", flush=True)
                continue
            return value
        except ValueError:
            print("Please enter an integer value.", flush=True)


def configure_program(io: SerialLineIO, *, power_w: float, frequency_khz: int) -> None:
    """Send the one-time program setup command before trial startup."""
    current_program = None
    try:
        current_program = io.get_program_values(timeout=2.0)
        log(
            "Current program values: "
            f"id={current_program['program_id']:02d}, "
            f"power={current_program['power_w']:04d}, "
            f"freq={current_program['frequency_khz']:04d}, "
            f"width={current_program['pulse_width_us']:04d}, "
            f"delay={current_program['detect_delay_us']:08d}"
        )
    except Exception as exc:
        log(f"Warning: Could not read current program values: {exc}. Sending requested program values directly.")

    ack = io.set_program_values(
        power_w=power_w,
        frequency_khz=frequency_khz,
        program_id=None,
        pulse_width_us=None,
        detect_delay_us=None,
        current_values=current_program,
        timeout=2.0,
    )
    ok_ack, _ = parse_ack(ack)
    if not ack.startswith("*"):
        raise RuntimeError(f"Unexpected SET_PROGRAM acknowledgment: {ack}")
    if not ok_ack:
        raise RuntimeError(f"SET_PROGRAM returned error code: {ack}")


def get_program_defaults(io: SerialLineIO, *, fallback_power_w: float, fallback_frequency_khz: int) -> tuple[float, int]:
    """Read the current program and use it to seed power/frequency defaults."""
    try:
        current_program = io.get_program_values(timeout=2.0)
        log(
            "Loaded startup defaults from hardware: "
            f"power={current_program['power_w']:04d}, "
            f"freq={current_program['frequency_khz']:04d}, "
            f"width={current_program['pulse_width_us']:04d}, "
            f"delay={current_program['detect_delay_us']:08d}"
        )
        return float(current_program["power_w"]), int(current_program["frequency_khz"])
    except Exception as exc:
        log(f"Warning: Could not load startup defaults from hardware: {exc}")
        return float(fallback_power_w), int(fallback_frequency_khz)


def reset_pid_defaults(io: SerialLineIO) -> None:
    """Write known-safe default PID values back to the controller."""
    ack = io.set_pid_values(
        pw_kp=0.15,
        pw_ki=0.14,
        pw_kd=0.05,
        pp_kp=0.15,
        pp_ki=0.14,
        pp_kd=0.05,
        holdoff=400.0,
        sample_interval=300.0,
        current_values=None,
        timeout=2.0,
    )
    ok_ack, _ = parse_ack(ack)
    if not ack.startswith("*"):
        raise RuntimeError(f"Unexpected reset acknowledgment: {ack}")
    if not ok_ack:
        raise RuntimeError(f"Reset returned error code: {ack}")


def format_readiness_status(
    *,
    bootstrap_status: dict | None,
    region_status: dict,
    safe_count: int,
    safe_target: int,
    good_count: int,
    good_target: int,
    warmup_trials_done: int | None = None,
) -> str:
    """Build a short GUI checklist for bootstrap readiness."""

    def mark(done: bool) -> str:
        return "[x]" if done else "[ ]"

    bootstrap_line = "Bootstrap: searching for stable local region"
    if bootstrap_status is not None:
        bootstrap_line = (
            f"Bootstrap: {int(warmup_trials_done or 0)} trials run | "
            f"floor {int(bootstrap_status.get('minimum_required', 0))} "
            f"{'reached' if bool(bootstrap_status.get('minimum_reached')) else 'not yet reached'} | "
            f"cap {int(bootstrap_status.get('hard_cap', 0))}"
        )

    center_pid = region_status.get("center_pid")
    if center_pid is None:
        center_text = "Local centre: waiting for best candidate"
    else:
        center_text = (
            "Local centre: "
            f"Kp={center_pid[0]:.4f}, Ki={center_pid[1]:.4f}, Kd={center_pid[2]:.4f}"
        )

    local_safe = int(region_status.get("local_safe_count", 0))
    local_good = int(region_status.get("local_good_count", 0))
    local_unsafe = int(region_status.get("local_unsafe_count", 0))
    local_variation_axes = int(region_status.get("local_variation_axes", 0))
    local_spans = tuple(region_status.get("local_spans", (0.0, 0.0, 0.0)))
    local_counts = tuple(region_status.get("local_unique_counts", (0, 0, 0)))

    lines = [
        "Bootstrap readiness:",
        bootstrap_line,
        f"{mark(safe_count >= safe_target)} Safe candidates: {safe_count}/{safe_target}",
        f"{mark(good_count >= good_target)} Good candidates: {good_count}/{good_target}",
        center_text,
        (
            "Local region: "
            f"safe={local_safe}, good={local_good}, unstable={local_unsafe}, "
            f"axes_with_variation={local_variation_axes}/3"
        ),
        (
            "Local spread: "
            f"Kp {local_counts[0]} pts / {local_spans[0]:.4f}, "
            f"Ki {local_counts[1]} pts / {local_spans[1]:.4f}, "
            f"Kd {local_counts[2]} pts / {local_spans[2]:.4f}"
        ),
    ]

    axis_statuses = list(region_status.get("axis_statuses", []))
    if axis_statuses:
        lines.append("Secondary diagnostic:")
        for status in axis_statuses:
            lines.append(
                f"{str(status.get('axis_name', '?')).upper()} global safe spread "
                f"{int(status.get('distinct_safe_values', 0))}/{int(status.get('required_distinct_values', 0))}, "
                f"span {float(status.get('safe_span', 0.0)):.4f}/{float(status.get('required_safe_span', 0.0)):.4f}"
            )

    if bool(bootstrap_status and bootstrap_status.get("ready")):
        lines.append("Bootstrap complete: switching to optimisation")
    else:
        lines.append(str(region_status.get("reason", "Bootstrap: searching for stable local region")))
    return "\n".join(lines)


def format_warmup_change_message(
    base_pid: tuple[float, float, float] | None,
    candidate_pid: tuple[float, float, float] | None,
    used_axis: int | None,
    candidate_delta: float,
) -> str:
    """Describe the current bootstrap move for the monitor."""
    if candidate_pid is None:
        return "Bootstrap change: waiting for first candidate"
    if base_pid is None or used_axis is None:
        return "Bootstrap change: baseline trial using current hardware PID"
    return (
        "Bootstrap change: "
        f"base=({base_pid[0]:.4f}, {base_pid[1]:.4f}, {base_pid[2]:.4f}) -> "
        f"{AXIS_NAMES[used_axis]} {candidate_delta:+.4f} -> "
        f"candidate=({candidate_pid[0]:.4f}, {candidate_pid[1]:.4f}, {candidate_pid[2]:.4f})"
    )


def format_previous_warmup_result_message(
    *,
    score: float,
    metrics: dict,
    per_test_meta: list[dict],
    cancelled_candidate: bool,
    cancel_reason: str,
    aborted: bool,
    baseline_score: float | None,
    safe_invalid_ratio: float,
    safe_oscillation_rate: float,
    good_score_factor: float,
) -> str:
    """Summarise whether the last bootstrap candidate passed and why it failed."""
    reasons: list[str] = []
    if cancelled_candidate:
        reasons.append(cancel_reason or "remaining repeats cancelled")
    if aborted:
        reasons.append("trial aborted by safety condition")

    invalid_ratio = float(metrics.get("invalid_ratio", 0.0))
    if invalid_ratio > float(safe_invalid_ratio):
        reasons.append(f"invalid ratio {invalid_ratio:.2f} > {float(safe_invalid_ratio):.2f}")

    oscillation_rate = float(metrics.get("oscillation_rate", 0.0))
    if oscillation_rate > float(safe_oscillation_rate):
        reasons.append(f"oscillation {oscillation_rate:.2f} > {float(safe_oscillation_rate):.2f}")

    if baseline_score is not None and baseline_score > 0:
        good_limit = float(baseline_score) * float(good_score_factor)
        if score > good_limit:
            reasons.append(f"score {score:.2f} > good threshold {good_limit:.2f}")

    tolerated = []
    for idx, meta in enumerate(per_test_meta, start=1):
        cats = list(meta.get("failure_categories", []))
        if cats and not meta.get("cancellation_decision"):
            tolerated.append(f"repeat {idx}: {', '.join(cats)} tolerated")
    if reasons:
        if tolerated:
            reasons.append("; ".join(tolerated[:2]))
        return "Previous bootstrap result: blocked - " + "; ".join(reasons)

    suffix = ""
    if tolerated:
        suffix = f" | {'; '.join(tolerated[:2])}"
    return (
        "Previous bootstrap result: usable - "
        f"score={score:.2f}, invalid={invalid_ratio:.2f}, osc={oscillation_rate:.2f}{suffix}"
    )


def format_phase_display(mode: str) -> str:
    if mode == "validation":
        return "VALIDATION"
    if mode.startswith("surrogate") or mode in {"bo", "fallback"}:
        return "OPTIMISATION (LOCAL REFINEMENT)"
    return "BOOTSTRAP"


def format_candidate_source(mode: str) -> str:
    if mode == "surrogate":
        return "surrogate (predicted best)"
    if mode == "surrogate_explore":
        return "surrogate (exploration)"
    if mode == "bo":
        return "coordinate refinement"
    if mode == "fallback":
        return "fallback (local refinement step)"
    if mode == "validation":
        return "validation"
    return "bootstrap"


def format_pid_delta(
    candidate_pid: tuple[float, float, float],
    reference_pid: tuple[float, float, float] | None,
) -> str:
    if reference_pid is None:
        return "dKp=+0.0000, dKi=+0.0000, dKd=+0.0000"
    return (
        f"dKp={candidate_pid[0] - reference_pid[0]:+.4f}, "
        f"dKi={candidate_pid[1] - reference_pid[1]:+.4f}, "
        f"dKd={candidate_pid[2] - reference_pid[2]:+.4f}"
    )


def format_candidate_reason(
    mode: str,
    *,
    best_pid: tuple[float, float, float] | None,
    candidate_pid: tuple[float, float, float] | None,
    used_axis: int | None = None,
    candidate_delta: float | None = None,
) -> str:
    if mode == "surrogate":
        return "Selected near best PID using surrogate model"
    if mode == "surrogate_explore":
        return "Exploring nearby candidate for local improvement"
    if mode == "bo":
        return "Selected inside the local refinement box for Bayesian local search"
    if mode == "validation":
        return "Re-testing the current best candidate for hold stability"
    if mode == "fallback" and best_pid is not None and candidate_pid is not None and used_axis is not None:
        direction = "upward" if float(candidate_delta or 0.0) >= 0.0 else "downward"
        return f"Refining {AXIS_NAMES[used_axis].upper()} slightly {direction} from best candidate"
    return "Searching for a stable local region"


def counted_trial_totals(*, mode: str, n_trials: int, validation_trials: int) -> tuple[int | None, int, int, int]:
    if mode == "validation":
        return int(validation_trials), 0, 0, 1
    if mode == "warmup":
        return None, 1, 0, 0
    return int(n_trials), 0, 1, 0


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Serial port (e.g. /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--iters", type=int, default=20, help="Number of tuning trials")
    ap.add_argument("--log-data", action="store_true", help="Log some DATA lines too (can be spammy)")
    ap.add_argument("--log-data-every", type=int, default=50, help="If --log-data, log every Nth DATA line")
    ap.add_argument("--kp-max", type=float, default=1.0, help="Upper limit for Kp search/clamp")
    ap.add_argument("--ki-max", type=float, default=1.0, help="Upper limit for Ki search/clamp")
    ap.add_argument("--kd-max", type=float, default=0.2, help="Upper limit for Kd search/clamp")
    ap.add_argument("--desired-output", type=float, default=525.0, help="Target output value for scoring")
    ap.add_argument("--w-start", type=float, default=0.25, help="Weight: start power error")
    ap.add_argument("--w-track", type=float, default=0.60, help="Weight: average tracking error before hold quality takes over")
    ap.add_argument("--w-dev", type=float, default=0.80, help="Weight: within-test deviation")
    ap.add_argument("--w-max", type=float, default=0.60, help="Weight: peak absolute error")
    ap.add_argument("--w-repeat", type=float, default=0.30, help="Weight: repeatability across repeated tests")
    ap.add_argument("--w-strict", type=float, default=6.0, help="Weight: settled +/-1%% violations")
    ap.add_argument("--w-osc", type=float, default=7.0, help="Weight: oscillation while holding the setpoint")
    ap.add_argument("--w-overshoot", type=float, default=3.0, help="Weight: percent overshoot")
    ap.add_argument("--w-settle", type=float, default=2.0, help="Weight: settling time")
    ap.add_argument("--w-rise", type=float, default=0.10, help="Weight: rise time")
    ap.add_argument("--w-steady", type=float, default=8.0, help="Weight: steady-state error")
    ap.add_argument("--w-iae", type=float, default=0.03, help="Weight: integral absolute error")
    ap.add_argument("--w-ise", type=float, default=0.005, help="Weight: integral squared error")
    ap.add_argument("--w-tolerance-time", type=float, default=2.5, help="Reward weight for time in tolerance band")
    ap.add_argument("--w-post-var", type=float, default=6.0, help="Weight: post-settle variance")
    ap.add_argument("--w-hold", type=float, default=5.0, help="Weight: explicit final hold-quality penalty")
    ap.add_argument("--invalid-penalty", type=float, default=800.0, help="Penalty multiplier for invalid tests")
    ap.add_argument("--startup-grace-s", type=float, default=2.0, help="Seconds to ignore startup overshoot")
    ap.add_argument("--settled-window-samples", type=int, default=5, help="Consecutive in-band samples to mark settled")
    ap.add_argument("--max-step-kp", type=float, default=0.15, help="Max per-trial change in Kp")
    ap.add_argument("--max-step-ki", type=float, default=0.15, help="Max per-trial change in Ki")
    ap.add_argument("--max-step-kd", type=float, default=0.05, help="Max per-trial change in Kd")
    ap.add_argument("--step-shrink-factor", type=float, default=0.85, help="Step multiplier on new best score")
    ap.add_argument("--step-growth-factor", type=float, default=1.05, help="Step multiplier when not improving")
    ap.add_argument(
        "--coordinate-warmup-trials",
        type=int,
        default=9,
        help="Minimum bootstrap coordinate trials before optimisation is allowed to start",
    )
    ap.add_argument(
        "--max-bootstrap-trials",
        type=int,
        default=27,
        help="Hard cap on bootstrap local-region discovery trials; these do not count toward --iters",
    )
    ap.add_argument(
        "--bayes-min-safe-trials",
        type=int,
        default=4,
        help="Minimum number of safe bootstrap trials required before optimisation starts",
    )
    ap.add_argument(
        "--bayes-region-min-points-per-axis",
        type=int,
        default=3,
        help="Diagnostic target for distinct safe values per PID axis during bootstrap",
    )
    ap.add_argument(
        "--bayes-region-min-good-candidates",
        type=int,
        default=2,
        help="Minimum number of stable, target-holding bootstrap candidates required before optimisation starts",
    )
    ap.add_argument(
        "--bayes-region-good-score-factor",
        type=float,
        default=1.05,
        help="Bootstrap candidate counts as good if its score is at most this multiple of the baseline score",
    )
    ap.add_argument("--bayes-region-min-span-kp", type=float, default=0.05, help="Diagnostic local span target for Kp")
    ap.add_argument("--bayes-region-min-span-ki", type=float, default=0.05, help="Diagnostic local span target for Ki")
    ap.add_argument("--bayes-region-min-span-kd", type=float, default=0.01, help="Diagnostic local span target for Kd")
    ap.add_argument(
        "--bayes-safe-invalid-ratio",
        type=float,
        default=0.20,
        help="Maximum invalid ratio a bootstrap trial can have and still count as safe",
    )
    ap.add_argument(
        "--bayes-safe-oscillation-rate",
        type=float,
        default=0.30,
        help="Maximum oscillation rate a bootstrap trial can have and still count as safe",
    )
    ap.add_argument(
        "--repeat-cancel-osc-threshold",
        type=float,
        default=0.80,
        help="Flag a repeat for oscillation when this rate is met or exceeded; two strikes cancel the candidate",
    )
    ap.add_argument(
        "--repeat-cancel-score-regression-pct",
        type=float,
        default=8.0,
        help="Flag a repeat for score regression when it degrades by at least this percentage versus prior valid repeats",
    )
    ap.add_argument(
        "--lock-growth-after-improve-pct",
        type=float,
        default=20.0,
        help="If best improvement >= this, stop increasing step sizes on misses",
    )
    ap.add_argument("--early-stop-patience", type=int, default=12, help="Stop after N non-improving trials")
    ap.add_argument("--retest-best-every", type=int, default=0, help="Every N trials, re-run current best PID for verification (0 disables)")
    ap.add_argument("--refine-activate-improve-pct", type=float, default=25.0, help="Enable local refinement bounds after this best-improvement percentage")
    ap.add_argument("--refine-radius-kp", type=float, default=0.2, help="Refinement radius around best Kp")
    ap.add_argument("--refine-radius-ki", type=float, default=0.2, help="Refinement radius around best Ki")
    ap.add_argument("--refine-radius-kd", type=float, default=0.05, help="Refinement radius around best Kd")
    ap.add_argument("--no-gui", action="store_true", help="Disable launch GUI and use console prompts")
    ap.add_argument("--power-csv", default="tuning_power_readings.csv", help="CSV file for graphing power readings")
    ap.add_argument("--test-duration-s", type=float, default=12.0, help="Seconds per individual laser test")
    ap.add_argument("--warmup-repeats", type=int, default=3, help="Number of repeated tests per candidate during bootstrap")
    ap.add_argument("--bo-repeats", type=int, default=5, help="Number of repeated tests per candidate during optimisation")
    ap.add_argument("--frequency-khz", type=int, default=0, help="Laser frequency in kHz for the startup program command")
    ap.add_argument(
        "--surrogate-model",
        choices=("extra_trees", "random_forest", "none"),
        default="extra_trees",
        help="Lightweight surrogate model used after bootstrap",
    )
    ap.add_argument("--surrogate-min-samples", type=int, default=8, help="Minimum completed candidates before surrogate-guided search is enabled")
    ap.add_argument("--surrogate-retrain-every", type=int, default=1, help="Refit the surrogate every N completed candidates")
    ap.add_argument("--surrogate-pool-size", type=int, default=36, help="Candidate pool size scored by the surrogate before each proposal")
    ap.add_argument("--surrogate-explore-prob", type=float, default=0.25, help="Exploration probability during surrogate-guided search")
    ap.add_argument("--surrogate-jitter-scale", type=float, default=1.25, help="Scale factor for surrogate proposal jitter around safe/best points")
    ap.add_argument(
        "--fallback-refinement-step-scale",
        type=float,
        default=0.35,
        help="Scale factor that shrinks optimisation coordinate steps for local refinement",
    )
    ap.add_argument("--bo-refine-trials", type=int, default=4, help="Optional BO-style refinement trials after surrogate search (0 disables)")
    ap.add_argument("--validation-trials", type=int, default=1, help="Optional validation re-tests of the final best candidate after optimisation (does not count toward --iters)")
    ap.add_argument("--validation-repeats", type=int, default=5, help="Number of repeated tests per validation candidate")
    args = ap.parse_args()

    default_goal = float(args.desired_output)
    default_frequency_khz = int(args.frequency_khz)
    if args.port:
        startup_ser = None
        try:
            startup_ser = serial.Serial(args.port, args.baud, timeout=0.1)
            startup_io = SerialLineIO(
                startup_ser,
                log_fn=log,
                log_data_lines=args.log_data,
                data_log_every=args.log_data_every,
            )
            default_goal, default_frequency_khz = get_program_defaults(
                startup_io,
                fallback_power_w=default_goal,
                fallback_frequency_khz=default_frequency_khz,
            )
        except Exception as exc:
            log(f"Warning: Could not open serial port for startup defaults: {exc}")
        finally:
            if startup_ser is not None:
                startup_ser.close()

    root = None
    while True:
        action = None
        desired_output = None
        n_trials = None
        test_duration_s = None
        frequency_khz = None
        monitor = None

        if not args.no_gui:
            ui = prompt_launch_gui(default_goal, args.iters, args.test_duration_s, default_frequency_khz, root=root)
            if ui is not None:
                root = ui.get("root")
                action = ui["action"]
                desired_output = float(ui["goal"])
                n_trials = int(ui["trials"])
                test_duration_s = float(ui["test_duration_s"])
                frequency_khz = int(ui["frequency_khz"])
                if action == "start" and root is not None:
                    monitor = RuntimeMonitor(root, desired_output=desired_output)
            else:
                log("GUI unavailable; falling back to console prompts.")

        if action is None:
            action = prompt_launch_action()
            if action == "start":
                desired_output = prompt_goal_power_output(default_goal)
                n_trials = prompt_trial_count(args.iters)
                frequency_khz = prompt_frequency_khz(default_frequency_khz)

        if action == "quit":
            log("Exiting on user request.")
            return
        if action == "graph":
            try:
                run_graph_tool(args.power_csv, prefer_gui=(not args.no_gui))
            except RuntimeError as exc:
                log(f"Graph tool error: {exc}")
            log("Graph tool closed; returning to main menu.")
            continue

        if not args.port:
            raise SystemExit("--port is required for start and reset actions")

        log("Opening serial port")
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        io = SerialLineIO(
            ser,
            log_fn=log,
            log_data_lines=args.log_data,
            data_log_every=args.log_data_every,
        )
        try:
            if action == "reset":
                log("Resetting PID values to defaults")
                reset_pid_defaults(io)
                log("Defaults restored successfully. Returning to main menu.")
                continue

            if desired_output is None:
                desired_output = prompt_goal_power_output(default_goal)
            log(f"Goal power output set to {desired_output:.4f}")
            if n_trials is None:
                n_trials = prompt_trial_count(args.iters)
            log(f"Configured counted optimisation trials after bootstrap: {n_trials}")
            if test_duration_s is None:
                test_duration_s = float(args.test_duration_s)
            log(f"Configured per-test duration: {test_duration_s:.2f}s")
            log(f"Configured repeats: warmup={int(args.warmup_repeats)}, BO={int(args.bo_repeats)}")
            if frequency_khz is None:
                frequency_khz = int(default_frequency_khz)
            log(f"Configured frequency: {frequency_khz} kHz")

            if monitor is not None:
                monitor.set_target(desired_output)
                monitor.set_phase("BOOTSTRAP")
                monitor.set_status("Sending startup program command")
                monitor.set_candidate_source("bootstrap")

            configure_program(io, power_w=desired_output, frequency_khz=frequency_khz)
            log(
                "Program setup sent: "
                f"power={int(round(desired_output)):04d}, "
                f"frequency={int(frequency_khz):04d} "
                "(other program fields preserved from GET_PROGRAM when available)"
            )
            if monitor is not None:
                monitor.set_status("Program setup applied")

            duration = 15.0
            history: list[dict] = []
            power_rows: list[tuple] = []
            trace_feature_rows: list[dict] = []
            trial_index = 0
            bootstrap_trial_count = 0
            optimisation_trial_count = 0
            validation_trial_count = 0
            surrogate_trial_count = 0
            bo_trial_count = 0
            fallback_trial_count = 0
            baseline_score = None
            best_score_seen = float("inf")
            best_pid = None
            best_metrics: dict | None = None
            last_applied = None
            no_improve_count = 0
            step_kp = float(args.max_step_kp)
            step_ki = float(args.max_step_ki)
            step_kd = float(args.max_step_kd)
            min_step_kp = max(0.01, step_kp * 0.1)
            min_step_ki = max(0.01, step_ki * 0.1)
            min_step_kd = max(0.005, step_kd * 0.1)
            axis_index = 0
            axis_directions = [1.0, 1.0, 1.0]
            safe_trial_points: list[tuple[float, float, float]] = []
            good_trial_points: list[tuple[float, float, float]] = []
            unsafe_trial_points: list[tuple[float, float, float]] = []
            observed_points: list[tuple[float, float, float]] = []
            observed_scores: list[float] = []
            surrogate_training_rows: list[dict] = []
            surrogate = OnlineSurrogateModel(args.surrogate_model, random_state=42)
            surrogate_active = False
            rng = random.Random(42)
            bootstrap_target = max(1, int(args.coordinate_warmup_trials))
            max_bootstrap_trials = max(bootstrap_target, int(args.max_bootstrap_trials))

            region_status = assess_local_safe_region(
                safe_trial_points,
                good_trial_points,
                unsafe_trial_points,
                best_pid=best_pid,
                min_safe_candidates=args.bayes_min_safe_trials,
                min_good_candidates=args.bayes_region_min_good_candidates,
                min_points_per_axis=args.bayes_region_min_points_per_axis,
                min_span_kp=args.bayes_region_min_span_kp,
                min_span_ki=args.bayes_region_min_span_ki,
                min_span_kd=args.bayes_region_min_span_kd,
                step_kp=step_kp,
                step_ki=step_ki,
                step_kd=step_kd,
            )
            bootstrap_status = assess_bootstrap_progress(
                bootstrap_trials_done=0,
                bootstrap_trials_minimum=bootstrap_target,
                max_bootstrap_trials=max_bootstrap_trials,
                region_status=region_status,
            )

            if monitor is not None:
                monitor.set_trial_counters(bootstrap_used=0, optimisation_used=0, validation_used=0)
                monitor.set_axis_coverage(region_status["axis_statuses"])
                monitor.set_best_candidate(kp=None, ki=None, kd=None, score=None)
                monitor.set_warmup_counter(
                    f"Bootstrap: 0 trials run | floor {bootstrap_target} not yet reached | cap {max_bootstrap_trials}"
                )
                monitor.set_warmup_change("Bootstrap change: baseline trial using current hardware PID")
                monitor.set_previous_warmup_result("Previous bootstrap result: none yet")
                monitor.set_readiness(
                    format_readiness_status(
                        bootstrap_status=bootstrap_status,
                        region_status=region_status,
                        safe_count=0,
                        safe_target=int(args.bayes_min_safe_trials),
                        good_count=0,
                        good_target=int(args.bayes_region_min_good_candidates),
                        warmup_trials_done=0,
                    )
                )

            def refinement_center_pid() -> tuple[float, float, float] | None:
                if best_pid is not None:
                    return tuple(float(v) for v in best_pid)
                if last_applied is not None:
                    return tuple(float(v) for v in last_applied)
                return None

            def local_refinement_bounds() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
                center_pid = refinement_center_pid()
                if center_pid is None:
                    return None
                return build_local_refinement_bounds(
                    center_pid,
                    kp_max=args.kp_max,
                    ki_max=args.ki_max,
                    kd_max=args.kd_max,
                    radius_kp=args.refine_radius_kp,
                    radius_ki=args.refine_radius_ki,
                    radius_kd=args.refine_radius_kd,
                )

            def local_refinement_step_sizes(
                current_step_kp: float,
                current_step_ki: float,
                current_step_kd: float,
            ) -> tuple[float, float, float]:
                return compute_refinement_step_sizes(
                    step_kp=current_step_kp,
                    step_ki=current_step_ki,
                    step_kd=current_step_kd,
                    radius_kp=args.refine_radius_kp,
                    radius_ki=args.refine_radius_ki,
                    radius_kd=args.refine_radius_kd,
                    scale=args.fallback_refinement_step_scale,
                )

            def local_refinement_step_floor() -> tuple[float, float, float]:
                return compute_refinement_step_sizes(
                    step_kp=min_step_kp,
                    step_ki=min_step_ki,
                    step_kd=min_step_kd,
                    radius_kp=args.refine_radius_kp,
                    radius_ki=args.refine_radius_ki,
                    radius_kd=args.refine_radius_kd,
                    scale=max(0.5, float(args.fallback_refinement_step_scale)),
                )

            def clamp_refinement_steps() -> None:
                nonlocal step_kp, step_ki, step_kd
                step_kp, step_ki, step_kd = local_refinement_step_sizes(step_kp, step_ki, step_kd)

            def mode_to_phase(mode: str) -> str:
                if mode == "validation":
                    return "validation"
                if mode == "warmup":
                    return "bootstrap"
                return "optimisation"

            def append_trial_logs(
                *,
                phase: str,
                mode: str,
                used_kp: float,
                used_ki: float,
                used_kd: float,
                score: float,
                metrics: dict,
                repeat_features: list[dict],
                per_test_powers: list[np.ndarray],
                per_test_times: list[np.ndarray],
                per_test_meta: list[dict],
                predicted_score: float | None,
                surrogate_enabled: bool,
                cancelled_candidate: bool,
                cancel_reason: str,
                aborted: bool,
                improve_vs_base_pct: float,
                best_improve_vs_base_pct: float,
                phase_trial_index: int,
            ) -> None:
                bootstrap_used = int(bootstrap_trial_count + (1 if phase == "bootstrap" else 0))
                optimisation_used = int(optimisation_trial_count + (1 if phase == "optimisation" else 0))
                validation_used = int(validation_trial_count + (1 if phase == "validation" else 0))
                history_row = {
                    "trial_index": int(trial_index + 1),
                    "phase": str(phase),
                    "phase_trial_index": int(phase_trial_index),
                    "bootstrap_trials_used": bootstrap_used,
                    "optimisation_trials_used": optimisation_used,
                    "validation_trials_used": validation_used,
                    "phase_mode": str(mode),
                    "candidate_selection_mode": str(mode),
                    "surrogate_active": int(bool(surrogate_enabled and surrogate_active)),
                    "predicted_score": float(predicted_score) if predicted_score is not None and np.isfinite(predicted_score) else math.nan,
                    "desired_output": float(desired_output),
                    "frequency_khz": int(frequency_khz),
                    "kp": float(used_kp),
                    "ki": float(used_ki),
                    "kd": float(used_kd),
                    "score": float(score),
                    "improve_vs_baseline_pct": float(improve_vs_base_pct),
                    "best_improve_vs_baseline_pct": float(best_improve_vs_base_pct),
                    "cancelled_repeats": int(cancelled_candidate),
                    "cancel_reason": str(cancel_reason),
                    "aborted": int(aborted),
                }
                history_row.update(metrics)
                history.append(history_row)

                for repeat_idx, (test_meta, repeat_feature) in enumerate(zip(per_test_meta, repeat_features), start=1):
                    trace_feature_rows.append(
                        {
                            "trial_index": int(trial_index + 1),
                            "phase": str(phase),
                            "phase_trial_index": int(phase_trial_index),
                            "bootstrap_trials_used": bootstrap_used,
                            "optimisation_trials_used": optimisation_used,
                            "validation_trials_used": validation_used,
                            "repeat_index": int(repeat_idx),
                            "phase_mode": str(mode),
                            "kp": float(used_kp),
                            "ki": float(used_ki),
                            "kd": float(used_kd),
                            "desired_output": float(desired_output),
                            "frequency_khz": int(frequency_khz),
                            "test_invalid": int(1 if bool(test_meta.get("invalid", False)) else 0),
                            "test_note": str(test_meta.get("reason", "")),
                            "strict_bad_rate": float(test_meta.get("strict_bad_rate", 1.0)),
                            "oscillation_rate": float(test_meta.get("oscillation_rate", 1.0)),
                            "failure_categories": "|".join(test_meta.get("failure_categories", [])),
                            "repeat_score_regression_pct": float(test_meta.get("score_regression_pct", 0.0)),
                            "repeat_cancellation_decision": str(test_meta.get("cancellation_decision", "")),
                            **repeat_feature,
                        }
                    )

                for test_idx, (test_powers, test_times, test_meta) in enumerate(zip(per_test_powers, per_test_times, per_test_meta), start=1):
                    if test_powers.size == 0:
                        continue
                    time_vals = test_times.tolist() if test_times.size == test_powers.size else list(range(int(test_powers.size)))
                    for sample_idx, (t_s, power_val) in enumerate(zip(time_vals, test_powers.tolist()), start=1):
                        power_rows.append(
                            (
                                int(trial_index + 1),
                                str(phase),
                                int(phase_trial_index),
                                bootstrap_used,
                                optimisation_used,
                                validation_used,
                                int(test_idx),
                                int(sample_idx),
                                float(t_s),
                                float(power_val),
                                float(desired_output),
                                float(used_kp),
                                float(used_ki),
                                float(used_kd),
                                int(1 if bool(test_meta.get("invalid", False)) else 0),
                                str(test_meta.get("reason", "")),
                            )
                        )

            def evaluate_candidate(
                kp: float,
                ki: float,
                kd: float,
                *,
                mode: str,
                used_axis: int | None = None,
                candidate_delta: float | None = None,
                predicted_score: float | None = None,
                surrogate_enabled: bool = False,
                candidate_reason: str | None = None,
            ) -> float:
                nonlocal trial_index, bootstrap_trial_count, optimisation_trial_count, validation_trial_count
                nonlocal surrogate_trial_count, bo_trial_count, fallback_trial_count
                nonlocal baseline_score, best_score_seen, best_pid, best_metrics, last_applied
                nonlocal no_improve_count, step_kp, step_ki, step_kd, axis_index, surrogate_active
                nonlocal region_status, bootstrap_status

                phase = mode_to_phase(mode)
                is_warmup_mode = mode == "warmup"
                is_surrogate_mode = mode.startswith("surrogate")
                is_bo_mode = mode == "bo"
                is_validation_mode = mode == "validation"
                is_optimisation_mode = phase == "optimisation"
                reference_best_pid = best_pid
                candidate_pid = (float(kp), float(ki), float(kd))
                selection_reason = candidate_reason or format_candidate_reason(
                    mode,
                    best_pid=reference_best_pid,
                    candidate_pid=candidate_pid,
                    used_axis=used_axis,
                    candidate_delta=candidate_delta,
                )

                if is_warmup_mode:
                    display_phase_name = "Bootstrap"
                    display_phase_index = bootstrap_trial_count + 1
                elif is_validation_mode:
                    display_phase_name = "Validation"
                    display_phase_index = validation_trial_count + 1
                elif is_surrogate_mode:
                    display_phase_name = "Optimisation"
                    display_phase_index = surrogate_trial_count + 1
                elif is_bo_mode:
                    display_phase_name = "Optimisation"
                    display_phase_index = bo_trial_count + 1
                else:
                    display_phase_name = "Optimisation"
                    display_phase_index = fallback_trial_count + 1

                display_phase_total, bootstrap_increment, optimisation_increment, validation_increment = counted_trial_totals(
                    mode=mode,
                    n_trials=int(n_trials),
                    validation_trials=int(args.validation_trials),
                )
                phase_repeats = max(
                    1,
                    int(args.warmup_repeats if is_warmup_mode else args.validation_repeats if is_validation_mode else args.bo_repeats),
                )
                log(
                    f"{display_phase_name} trial {display_phase_index}"
                    + (f"/{display_phase_total}" if display_phase_total is not None else "")
                    + f" (overall {trial_index + 1})"
                )
                log(
                    "Candidate selection -> "
                    f"source={format_candidate_source(mode)}, "
                    f"reason={selection_reason}, "
                    f"{format_pid_delta(candidate_pid, reference_best_pid)}"
                )

                if monitor is not None:
                    monitor.set_phase(format_phase_display(mode))
                    monitor.set_candidate_source(format_candidate_source(mode))
                    monitor.set_candidate_reason(selection_reason)
                    monitor.set_trial_counters(
                        bootstrap_used=bootstrap_trial_count + bootstrap_increment,
                        optimisation_used=optimisation_trial_count + optimisation_increment,
                        validation_used=validation_trial_count + validation_increment,
                    )
                    if is_surrogate_mode:
                        monitor.set_warmup_change(
                            (
                                f"Optimisation proposal: predicted score {predicted_score:.2f}"
                                if predicted_score is not None and np.isfinite(predicted_score)
                                else "Optimisation proposal: model-guided candidate"
                            )
                        )
                    elif is_bo_mode:
                        monitor.set_warmup_change("Optimisation change: Bayesian refinement active")
                    elif is_validation_mode:
                        monitor.set_warmup_change("Validation change: re-testing best candidate hold quality")
                    elif mode == "fallback":
                        monitor.set_warmup_change("Optimisation change: fallback coordinate search")
                    else:
                        monitor.set_warmup_change("Bootstrap: searching for stable local region")
                    progress = f"{display_phase_name} trial {display_phase_index}"
                    if display_phase_total is not None:
                        progress = f"{progress}/{display_phase_total}"
                    monitor.set_progress(f"{progress} | configuring hardware | overall {trial_index + 1}")

                apply_pid_update = trial_index > 0
                if not apply_pid_update:
                    kp, ki, kd = 0.0, 0.0, 0.0
                    log("Baseline trial -> using current laser PID values without changing gains")
                else:
                    log(f"{display_phase_name} candidate -> kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

                _, _, _, aborted, current_pid, per_test_powers, per_test_times, per_test_meta, cancelled_candidate, cancel_reason = run_trial(
                    io,
                    kp,
                    ki,
                    kd,
                    desired_output=desired_output,
                    apply_pid_update=apply_pid_update,
                    repeats=phase_repeats,
                    test_duration_s=test_duration_s,
                    startup_grace_s=args.startup_grace_s,
                    settled_window_samples=args.settled_window_samples,
                    duration=duration,
                    kp_max=args.kp_max,
                    ki_max=args.ki_max,
                    kd_max=args.kd_max,
                    monitor=monitor,
                    trial_index=trial_index + 1,
                    phase_name=display_phase_name,
                    phase_trial_index=display_phase_index,
                    phase_trial_total=display_phase_total,
                    overall_trial_index=trial_index + 1,
                    best_pid=reference_best_pid,
                    repeat_cancel_osc_threshold=args.repeat_cancel_osc_threshold,
                    repeat_cancel_score_regression_pct=args.repeat_cancel_score_regression_pct,
                )

                used_kp, used_ki, used_kd = kp, ki, kd
                if trial_index == 0 and current_pid is not None:
                    used_kp = float(current_pid["pw_kp"])
                    used_ki = float(current_pid["pw_ki"])
                    used_kd = float(current_pid["pw_kd"])
                    log(
                        "Stored initial laser PID values for baseline trial: "
                        f"kp={used_kp:.4f}, ki={used_ki:.4f}, kd={used_kd:.4f}"
                    )
                    last_applied = (used_kp, used_ki, used_kd)
                elif trial_index == 0:
                    last_applied = None
                else:
                    last_applied = (used_kp, used_ki, used_kd)

                metrics, repeat_features = compute_trial_metrics(
                    per_test_powers,
                    per_test_times,
                    per_test_meta,
                    desired_output,
                    settled_window_samples=args.settled_window_samples,
                )
                score = score_controller(
                    metrics,
                    w_start=args.w_start,
                    w_track=args.w_track,
                    w_dev=args.w_dev,
                    w_max=args.w_max,
                    w_repeat=args.w_repeat,
                    w_strict=args.w_strict,
                    w_osc=args.w_osc,
                    w_overshoot=args.w_overshoot,
                    w_settle=args.w_settle,
                    w_rise=args.w_rise,
                    w_steady=args.w_steady,
                    w_iae=args.w_iae,
                    w_ise=args.w_ise,
                    w_tolerance_time=args.w_tolerance_time,
                    w_post_var=args.w_post_var,
                    w_hold=args.w_hold,
                    invalid_penalty=args.invalid_penalty,
                    cancelled_candidate=cancelled_candidate,
                    aborted=aborted,
                )

                prev_best_score = best_score_seen
                if baseline_score is None:
                    baseline_score = score
                improved = score < prev_best_score
                if not is_validation_mode:
                    best_score_seen = min(best_score_seen, score)
                    local_floor_kp, local_floor_ki, local_floor_kd = local_refinement_step_floor()
                    if improved:
                        best_pid = (used_kp, used_ki, used_kd)
                        best_metrics = dict(metrics)
                        no_improve_count = 0
                        if is_optimisation_mode:
                            step_kp = max(local_floor_kp, step_kp * float(args.step_shrink_factor))
                            step_ki = max(local_floor_ki, step_ki * float(args.step_shrink_factor))
                            step_kd = max(local_floor_kd, step_kd * float(args.step_shrink_factor))
                            clamp_refinement_steps()
                        else:
                            step_kp = max(min_step_kp, step_kp * float(args.step_shrink_factor))
                            step_ki = max(min_step_ki, step_ki * float(args.step_shrink_factor))
                            step_kd = max(min_step_kd, step_kd * float(args.step_shrink_factor))
                    else:
                        no_improve_count += 1
                        if used_axis is not None:
                            axis_directions[used_axis] *= -1.0
                        if is_optimisation_mode:
                            step_kp = max(local_floor_kp, step_kp * float(args.step_shrink_factor))
                            step_ki = max(local_floor_ki, step_ki * float(args.step_shrink_factor))
                            step_kd = max(local_floor_kd, step_kd * float(args.step_shrink_factor))
                            clamp_refinement_steps()
                        else:
                            if baseline_score is not None and baseline_score > 0:
                                best_improve_pct_so_far = 100.0 * (baseline_score - best_score_seen) / baseline_score
                            else:
                                best_improve_pct_so_far = 0.0
                            if best_improve_pct_so_far < float(args.lock_growth_after_improve_pct):
                                step_kp = min(float(args.max_step_kp), step_kp * float(args.step_growth_factor))
                                step_ki = min(float(args.max_step_ki), step_ki * float(args.step_growth_factor))
                                step_kd = min(float(args.max_step_kd), step_kd * float(args.step_growth_factor))
                    if used_axis is not None:
                        axis_index = (used_axis + 1) % 3

                if baseline_score and baseline_score > 0:
                    improve_vs_base_pct = 100.0 * (baseline_score - score) / baseline_score
                    best_improve_vs_base_pct = 100.0 * (baseline_score - best_score_seen) / baseline_score
                else:
                    improve_vs_base_pct = 0.0
                    best_improve_vs_base_pct = 0.0

                if apply_pid_update and not is_validation_mode:
                    observed_points.append((used_kp, used_ki, used_kd))
                    observed_scores.append(score)
                    surrogate_training_rows.append(
                        build_surrogate_training_row(
                            used_kp,
                            used_ki,
                            used_kd,
                            desired_output=desired_output,
                            frequency_khz=frequency_khz,
                            score=score,
                        )
                    )
                    is_safe_candidate = candidate_is_safe(
                        metrics,
                        cancelled_candidate=cancelled_candidate,
                        aborted=aborted,
                        max_invalid_ratio=args.bayes_safe_invalid_ratio,
                        max_oscillation_rate=args.bayes_safe_oscillation_rate,
                    )
                    if is_safe_candidate:
                        safe_trial_points.append((used_kp, used_ki, used_kd))
                    else:
                        unsafe_trial_points.append((used_kp, used_ki, used_kd))
                    if candidate_is_good(
                        metrics,
                        score,
                        cancelled_candidate=cancelled_candidate,
                        aborted=aborted,
                        baseline_score=baseline_score,
                        max_invalid_ratio=args.bayes_safe_invalid_ratio,
                        max_oscillation_rate=args.bayes_safe_oscillation_rate,
                        max_score_factor=args.bayes_region_good_score_factor,
                    ):
                        good_trial_points.append((used_kp, used_ki, used_kd))

                    region_status = assess_local_safe_region(
                        safe_trial_points,
                        good_trial_points,
                        unsafe_trial_points,
                        best_pid=best_pid,
                        min_safe_candidates=args.bayes_min_safe_trials,
                        min_good_candidates=args.bayes_region_min_good_candidates,
                        min_points_per_axis=args.bayes_region_min_points_per_axis,
                        min_span_kp=args.bayes_region_min_span_kp,
                        min_span_ki=args.bayes_region_min_span_ki,
                        min_span_kd=args.bayes_region_min_span_kd,
                        step_kp=step_kp,
                        step_ki=step_ki,
                        step_kd=step_kd,
                    )
                    bootstrap_done = bootstrap_trial_count + 1
                    bootstrap_status = assess_bootstrap_progress(
                        bootstrap_trials_done=bootstrap_done,
                        bootstrap_trials_minimum=bootstrap_target,
                        max_bootstrap_trials=max_bootstrap_trials,
                        region_status=region_status,
                    )
                    log(
                        "Bootstrap region status -> "
                        f"region_ready={region_status['ready']}, "
                        f"bootstrap_ready={bootstrap_status['ready']}, "
                        f"reason={bootstrap_status['reason']}, "
                        f"local_safe={region_status.get('local_safe_count', 0)}, "
                        f"local_good={region_status.get('local_good_count', 0)}, "
                        f"local_unsafe={region_status.get('local_unsafe_count', 0)}, "
                        f"variation_axes={region_status.get('local_variation_axes', 0)}/3"
                    )
                    if monitor is not None:
                        monitor.set_axis_coverage(region_status["axis_statuses"])
                        monitor.set_warmup_counter(
                            f"Bootstrap: {bootstrap_done} trials run | "
                            f"floor {bootstrap_target} "
                            f"{'reached' if bootstrap_status['minimum_reached'] else 'not yet reached'} | "
                            f"cap {max_bootstrap_trials}"
                        )
                        if is_warmup_mode:
                            monitor.set_previous_warmup_result(
                                format_previous_warmup_result_message(
                                    score=score,
                                    metrics=metrics,
                                    per_test_meta=per_test_meta,
                                    cancelled_candidate=cancelled_candidate,
                                    cancel_reason=cancel_reason,
                                    aborted=aborted,
                                    baseline_score=baseline_score,
                                    safe_invalid_ratio=args.bayes_safe_invalid_ratio,
                                    safe_oscillation_rate=args.bayes_safe_oscillation_rate,
                                    good_score_factor=args.bayes_region_good_score_factor,
                                )
                            )
                        monitor.set_readiness(
                            format_readiness_status(
                                bootstrap_status=bootstrap_status,
                                region_status=region_status,
                                safe_count=len(safe_trial_points),
                                safe_target=int(args.bayes_min_safe_trials),
                                good_count=len(good_trial_points),
                                good_target=int(args.bayes_region_min_good_candidates),
                                warmup_trials_done=bootstrap_done,
                            )
                        )
                        if bool(bootstrap_status.get("ready")):
                            monitor.set_progress("Local safe region found near best candidate")
                            monitor.set_warmup_change("Bootstrap change: complete")

                    if args.surrogate_model != "none" and (
                        len(surrogate_training_rows) % max(1, int(args.surrogate_retrain_every)) == 0
                    ):
                        surrogate_active = surrogate.fit(
                            surrogate_training_rows,
                            min_samples=max(int(args.surrogate_min_samples), int(args.bayes_min_safe_trials)),
                        )
                    elif len(surrogate_training_rows) < int(args.surrogate_min_samples):
                        surrogate_active = False

                if monitor is not None:
                    monitor.set_trial_counters(
                        bootstrap_used=bootstrap_trial_count + bootstrap_increment,
                        optimisation_used=optimisation_trial_count + optimisation_increment,
                        validation_used=validation_trial_count + validation_increment,
                    )
                    monitor.set_axis_coverage(region_status["axis_statuses"])
                    if best_pid is not None:
                        best_metrics_payload = best_metrics or metrics
                        monitor.set_best_candidate(
                            kp=best_pid[0],
                            ki=best_pid[1],
                            kd=best_pid[2],
                            score=best_score_seen,
                            overshoot_pct=best_metrics_payload.get("overshoot_pct"),
                            hold_quality=best_metrics_payload.get("hold_quality"),
                        )

                append_trial_logs(
                    phase=phase,
                    mode=mode,
                    used_kp=used_kp,
                    used_ki=used_ki,
                    used_kd=used_kd,
                    score=score,
                    metrics=metrics,
                    repeat_features=repeat_features,
                    per_test_powers=per_test_powers,
                    per_test_times=per_test_times,
                    per_test_meta=per_test_meta,
                    predicted_score=predicted_score,
                    surrogate_enabled=surrogate_enabled,
                    cancelled_candidate=cancelled_candidate,
                    cancel_reason=cancel_reason,
                    aborted=aborted,
                    improve_vs_base_pct=improve_vs_base_pct,
                    best_improve_vs_base_pct=best_improve_vs_base_pct,
                    phase_trial_index=display_phase_index,
                )

                log(
                    f"Result -> score={score:.2f}, improve={improve_vs_base_pct:.2f}%, "
                    f"best_improve={best_improve_vs_base_pct:.2f}%, no_improve={no_improve_count}, "
                    f"step=({step_kp:.4f},{step_ki:.4f},{step_kd:.4f}), "
                    f"track_err={metrics['track_error']:.5f}, max_err={metrics['max_error']:.5f}, "
                    f"osc={metrics['oscillation_rate']:.5f}, hold={metrics.get('hold_quality', math.nan):.3f}, "
                    f"invalid={metrics['invalid_ratio']:.3f}, repeat={metrics['repeatability']:.5f}, "
                    f"cancelled={cancelled_candidate}, aborted={aborted}"
                )
                if cancelled_candidate:
                    log(f"Candidate repeats cancelled early: {cancel_reason}")

                trial_index += 1
                if is_warmup_mode:
                    bootstrap_trial_count += 1
                elif is_validation_mode:
                    validation_trial_count += 1
                elif is_surrogate_mode:
                    optimisation_trial_count += 1
                    surrogate_trial_count += 1
                elif is_bo_mode:
                    optimisation_trial_count += 1
                    bo_trial_count += 1
                else:
                    optimisation_trial_count += 1
                    fallback_trial_count += 1

                if (
                    args.retest_best_every > 0
                    and phase == "optimisation"
                    and best_pid is not None
                    and (optimisation_trial_count % int(args.retest_best_every) == 0)
                ):
                    log(
                        "Periodic validation re-test -> "
                        f"kp={best_pid[0]:.4f}, ki={best_pid[1]:.4f}, kd={best_pid[2]:.4f}"
                    )
                    evaluate_candidate(best_pid[0], best_pid[1], best_pid[2], mode="validation")

                if (not is_validation_mode) and args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                    raise EarlyStopOptimization(
                        f"No score improvement for {no_improve_count} trials (patience={args.early_stop_patience})"
                    )
                return score

            def run_coordinate_trial() -> float:
                nonlocal axis_index
                base_pid = best_pid if best_pid is not None else last_applied
                used_axis = None
                candidate_delta = 0.0
                if trial_index > 0:
                    if base_pid is None:
                        base_pid = (0.0, 0.0, 0.0)
                    selected_axis, axis_statuses = choose_bootstrap_axis(
                        base_pid,
                        safe_points=safe_trial_points,
                        axis_directions=axis_directions,
                        preferred_axis_index=axis_index,
                        step_kp=step_kp,
                        step_ki=step_ki,
                        step_kd=step_kd,
                        kp_max=args.kp_max,
                        ki_max=args.ki_max,
                        kd_max=args.kd_max,
                        min_points_per_axis=args.bayes_region_min_points_per_axis,
                        min_span_kp=args.bayes_region_min_span_kp,
                        min_span_ki=args.bayes_region_min_span_ki,
                        min_span_kd=args.bayes_region_min_span_kd,
                    )
                    (kp, ki, kd), used_axis, _, candidate_delta = propose_coordinate_candidate(
                        base_pid,
                        axis_index=selected_axis,
                        axis_direction=axis_directions[selected_axis % 3],
                        step_kp=step_kp,
                        step_ki=step_ki,
                        step_kd=step_kd,
                        kp_max=args.kp_max,
                        ki_max=args.ki_max,
                        kd_max=args.kd_max,
                    )
                    axis_state = next((status for status in axis_statuses if int(status["axis_index"]) == int(used_axis)), None)
                    diagnostic = ""
                    if axis_state is not None:
                        diagnostic = (
                            f"global_safe={axis_state['distinct_safe_values']}/{axis_state['required_distinct_values']}, "
                            f"span={axis_state['safe_span']:.4f}/{axis_state['required_safe_span']:.4f}, "
                        )
                    log(
                        "Bootstrap candidate -> "
                        f"base=({base_pid[0]:.4f},{base_pid[1]:.4f},{base_pid[2]:.4f}), "
                        f"axis={AXIS_NAMES[used_axis]}, delta={candidate_delta:+.4f}, "
                        f"{diagnostic}candidate=({kp:.4f},{ki:.4f},{kd:.4f})"
                    )
                    if monitor is not None:
                        monitor.set_warmup_change(
                            format_warmup_change_message(base_pid, (kp, ki, kd), used_axis, candidate_delta)
                        )
                else:
                    kp, ki, kd = 0.0, 0.0, 0.0
                    if monitor is not None:
                        monitor.set_warmup_change(format_warmup_change_message(None, (kp, ki, kd), None, 0.0))

                score = evaluate_candidate(kp, ki, kd, mode="warmup", used_axis=used_axis)
                if used_axis is not None:
                    axis_index = (used_axis + 1) % 3
                return score

            def run_surrogate_trial() -> float:
                bounds = local_refinement_bounds()
                local_step_kp, local_step_ki, local_step_kd = local_refinement_step_sizes(step_kp, step_ki, step_kd)
                candidate, predicted, proposal_mode = propose_surrogate_candidate(
                    surrogate,
                    best_pid=best_pid,
                    safe_points=safe_trial_points,
                    observed_points=observed_points,
                    desired_output=desired_output,
                    frequency_khz=frequency_khz,
                    rng=rng,
                    pool_size=args.surrogate_pool_size,
                    explore_prob=args.surrogate_explore_prob,
                    jitter_scale=args.surrogate_jitter_scale,
                    step_kp=local_step_kp,
                    step_ki=local_step_ki,
                    step_kd=local_step_kd,
                    kp_max=args.kp_max,
                    ki_max=args.ki_max,
                    kd_max=args.kd_max,
                    local_bounds=bounds,
                )
                return evaluate_candidate(
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    mode=proposal_mode,
                    predicted_score=predicted,
                    surrogate_enabled=True,
                )

            def run_fallback_trial() -> float:
                nonlocal axis_index
                bounds = local_refinement_bounds()
                base_pid = best_pid if best_pid is not None else last_applied
                if base_pid is None:
                    base_pid = (0.0, 0.0, 0.0)
                base_pid = clamp_pid_to_bounds(base_pid, bounds)
                local_step_kp, local_step_ki, local_step_kd = local_refinement_step_sizes(step_kp, step_ki, step_kd)
                (kp, ki, kd), used_axis, _, candidate_delta = propose_coordinate_candidate(
                    base_pid,
                    axis_index=axis_index,
                    axis_direction=axis_directions[axis_index % 3],
                    step_kp=local_step_kp,
                    step_ki=local_step_ki,
                    step_kd=local_step_kd,
                    kp_max=args.kp_max,
                    ki_max=args.ki_max,
                    kd_max=args.kd_max,
                    bounds=bounds,
                )
                log(
                    "Fallback candidate -> "
                    f"base=({base_pid[0]:.4f},{base_pid[1]:.4f},{base_pid[2]:.4f}), "
                    f"axis={AXIS_NAMES[used_axis]}, delta={candidate_delta:+.4f}, "
                    f"candidate=({kp:.4f},{ki:.4f},{kd:.4f})"
                )
                if monitor is not None:
                    monitor.set_warmup_change(
                        "Fallback change: "
                        f"base=({base_pid[0]:.4f}, {base_pid[1]:.4f}, {base_pid[2]:.4f}) -> "
                        f"{AXIS_NAMES[used_axis]} {candidate_delta:+.4f} -> "
                        f"candidate=({kp:.4f}, {ki:.4f}, {kd:.4f})"
                    )
                score = evaluate_candidate(
                    kp,
                    ki,
                    kd,
                    mode="fallback",
                    used_axis=used_axis,
                    candidate_delta=candidate_delta,
                )
                axis_index = (used_axis + 1) % 3
                return score

            log("Starting bootstrap search for a stable local safe region")
            if monitor is not None:
                monitor.set_phase(format_phase_display("warmup"))
                monitor.set_candidate_source(format_candidate_source("warmup"))

            try:
                while bootstrap_trial_count < max_bootstrap_trials and not bool(bootstrap_status["ready"]):
                    run_coordinate_trial()

                if bool(bootstrap_status["ready"]):
                    no_improve_count = 0
                    clamp_refinement_steps()
                    surrogate_budget = max(0, int(n_trials) - max(0, int(args.bo_refine_trials)))
                    bo_budget = max(0, min(int(args.bo_refine_trials), int(n_trials)))
                    refinement_bounds = local_refinement_bounds()
                    log(
                        f"Bootstrap complete after {bootstrap_trial_count} trials; "
                        f"planned phase budgets -> surrogate={surrogate_budget}, bo_refine={bo_budget}"
                    )
                    if refinement_bounds is not None:
                        log(
                            "Entering optimisation phase -> "
                            f"local bounds Kp=[{refinement_bounds[0][0]:.4f}, {refinement_bounds[0][1]:.4f}], "
                            f"Ki=[{refinement_bounds[1][0]:.4f}, {refinement_bounds[1][1]:.4f}], "
                            f"Kd=[{refinement_bounds[2][0]:.4f}, {refinement_bounds[2][1]:.4f}], "
                            f"step=({step_kp:.4f},{step_ki:.4f},{step_kd:.4f})"
                        )
                    if monitor is not None:
                        monitor.set_phase(format_phase_display("surrogate"))
                        monitor.set_status("Local stable region found — refining around best candidate")
                        monitor.set_candidate_source("local refinement")
                        monitor.set_candidate_reason("Optimisation is now constrained around the best safe PID")
                        monitor.set_readiness(
                            format_readiness_status(
                                bootstrap_status=bootstrap_status,
                                region_status=region_status,
                                safe_count=len(safe_trial_points),
                                safe_target=int(args.bayes_min_safe_trials),
                                good_count=len(good_trial_points),
                                good_target=int(args.bayes_region_min_good_candidates),
                                warmup_trials_done=bootstrap_trial_count,
                            )
                        )

                    if args.surrogate_model != "none":
                        surrogate_active = surrogate.fit(
                            surrogate_training_rows,
                            min_samples=max(int(args.surrogate_min_samples), int(args.bayes_min_safe_trials)),
                        )
                        if surrogate_active:
                            log(f"Surrogate ready -> model={args.surrogate_model}, samples={surrogate.last_fit_count}")
                        else:
                            reason = surrogate.last_error or "not enough data"
                            log(f"Surrogate unavailable, falling back safely: {reason}")

                    optimisation_trials_run = 0
                    if surrogate_active and surrogate_budget > 0:
                        if monitor is not None:
                            monitor.set_phase(format_phase_display("surrogate"))
                            monitor.set_candidate_source(format_candidate_source("surrogate"))
                        while optimisation_trials_run < surrogate_budget:
                            run_surrogate_trial()
                            optimisation_trials_run += 1
                    elif not surrogate_active:
                        log("Surrogate phase skipped: model unavailable or insufficient data")

                    remaining_after_surrogate = max(0, int(n_trials) - optimisation_trials_run)
                    if remaining_after_surrogate > 0 and bo_budget > 0 and HAVE_SKOPT:
                        bo_calls = min(bo_budget, remaining_after_surrogate)
                        log(f"Starting BO refinement for {bo_calls} trial(s)")
                        if monitor is not None:
                            monitor.set_phase(format_phase_display("bo"))
                            monitor.set_candidate_source(format_candidate_source("bo"))
                        bayes_space = build_bayes_search_space(
                            safe_trial_points,
                            kp_max=args.kp_max,
                            ki_max=args.ki_max,
                            kd_max=args.kd_max,
                            pad_kp=step_kp,
                            pad_ki=step_ki,
                            pad_kd=step_kd,
                            best_pid=refinement_center_pid(),
                            refine_radius_kp=args.refine_radius_kp,
                            refine_radius_ki=args.refine_radius_ki,
                            refine_radius_kd=args.refine_radius_kd,
                        )
                        seed_points, seed_scores = filter_seed_points_for_space(observed_points, observed_scores, bayes_space)
                        gp_minimize(
                            lambda x: evaluate_candidate(
                                float(x[0]),
                                float(x[1]),
                                float(x[2]),
                                mode="bo",
                                candidate_reason="Selected inside the local refinement box for Bayesian local search",
                            ),
                            bayes_space,
                            n_calls=bo_calls,
                            n_initial_points=min(4, max(1, bo_calls)),
                            acq_func="EI",
                            random_state=42,
                            x0=seed_points if seed_points else None,
                            y0=seed_scores if seed_scores else None,
                        )
                        optimisation_trials_run += bo_calls
                    elif remaining_after_surrogate > 0 and bo_budget > 0 and not HAVE_SKOPT:
                        log("BO refinement unavailable because scikit-optimize is not installed")

                    remaining_after_bo = max(0, int(n_trials) - optimisation_trials_run)
                    while remaining_after_bo > 0:
                        log("Using local fallback refinement for remaining optimisation budget")
                        run_fallback_trial()
                        remaining_after_bo -= 1
                else:
                    log(
                        "Optimisation phases skipped: "
                        f"no stable local safe region found after {bootstrap_trial_count} bootstrap trials "
                        f"(floor {bootstrap_target}, cap {max_bootstrap_trials}; {bootstrap_status['reason']})"
                    )
                    if monitor is not None:
                        monitor.set_progress(
                            f"Bootstrap stopped after {bootstrap_trial_count} trials | {bootstrap_status['reason']}"
                        )
                        monitor.set_readiness(
                            format_readiness_status(
                                bootstrap_status=bootstrap_status,
                                region_status=region_status,
                                safe_count=len(safe_trial_points),
                                safe_target=int(args.bayes_min_safe_trials),
                                good_count=len(good_trial_points),
                                good_target=int(args.bayes_region_min_good_candidates),
                                warmup_trials_done=bootstrap_trial_count,
                            )
                        )
            except EarlyStopOptimization as exc:
                log(f"Early stop: {exc}")

            if bool(region_status["ready"]) and best_pid is not None and int(args.validation_trials) > 0 and optimisation_trial_count > 0:
                log(f"Starting validation phase for {int(args.validation_trials)} trial(s)")
                if monitor is not None:
                    monitor.set_phase(format_phase_display("validation"))
                for _ in range(int(args.validation_trials)):
                    evaluate_candidate(best_pid[0], best_pid[1], best_pid[2], mode="validation")

            if best_pid is not None:
                best_kp, best_ki, best_kd = best_pid
            elif last_applied is not None:
                best_kp, best_ki, best_kd = last_applied
            else:
                best_kp, best_ki, best_kd = 0.0, 0.0, 0.0

            log("Tuning complete")
            if monitor is not None:
                monitor.set_phase(format_phase_display("validation" if validation_trial_count > 0 else "fallback"))
                monitor.set_trial_counters(
                    bootstrap_used=bootstrap_trial_count,
                    optimisation_used=optimisation_trial_count,
                    validation_used=validation_trial_count,
                )
                monitor.set_axis_coverage(region_status["axis_statuses"])
                if best_pid is not None and best_metrics is not None:
                    monitor.set_best_candidate(
                        kp=best_pid[0],
                        ki=best_pid[1],
                        kd=best_pid[2],
                        score=best_score_seen,
                        overshoot_pct=best_metrics.get("overshoot_pct"),
                        hold_quality=best_metrics.get("hold_quality"),
                    )
            log(f"BEST kp={best_kp:.6f}, ki={best_ki:.6f}, kd={best_kd:.6f}")
            if best_score_seen < float('inf'):
                log(f"Best score={best_score_seen:.3f}")
            log(
                "Phase counts -> "
                f"bootstrap={bootstrap_trial_count}, optimisation={optimisation_trial_count}, validation={validation_trial_count}"
            )

            with open("tuning_history.csv", "w", newline="") as f:
                fieldnames = ordered_row_fieldnames(
                    history,
                    [
                        "trial_index",
                        "phase",
                        "phase_mode",
                        "candidate_selection_mode",
                        "surrogate_active",
                        "predicted_score",
                        "desired_output",
                        "frequency_khz",
                        "kp",
                        "ki",
                        "kd",
                        "score",
                    ],
                )
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(history)
            log("Saved tuning_history.csv")

            with open("tuning_trace_features.csv", "w", newline="") as f:
                fieldnames = ordered_row_fieldnames(
                    trace_feature_rows,
                    [
                        "trial_index",
                        "phase",
                        "repeat_index",
                        "phase_mode",
                        "kp",
                        "ki",
                        "kd",
                    ],
                )
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trace_feature_rows)
            log("Saved tuning_trace_features.csv")

            with open("tuning_power_readings.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "trial_index",
                        "phase",
                        "phase_trial_index",
                        "bootstrap_trials_used",
                        "optimisation_trials_used",
                        "validation_trials_used",
                        "test_index",
                        "sample_index",
                        "time_s",
                        "current_power",
                        "desired_output",
                        "kp",
                        "ki",
                        "kd",
                        "test_invalid",
                        "test_note",
                    ]
                )
                writer.writerows(power_rows)
            log("Saved tuning_power_readings.csv")
            default_goal = float(desired_output)
            default_frequency_khz = int(frequency_khz)
            if monitor is not None:
                monitor.mark_complete("Optimisation complete. Returning to main menu.")
        finally:
            ser.close()


if __name__ == "__main__":
    main()

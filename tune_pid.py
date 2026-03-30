"""Main application entry point for laser PID tuning.

This script provides:
- startup UI (GUI/CLI) for running actions
- serial command orchestration for each laser test
- trial scoring and hybrid PID tuning
- CSV output for later analysis
- interactive graphing of stored power traces
"""

import csv
from datetime import datetime
import math
import random
from typing import Any

import numpy as np
import serial
try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAVE_SKOPT = True
except Exception:
    gp_minimize = None
    Real = None
    HAVE_SKOPT = False

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    HAVE_SKLEARN = True
except Exception:
    ExtraTreesRegressor = None
    RandomForestRegressor = None
    HAVE_SKLEARN = False

import laser_command_ids as CMD
from pipeline.data_collector import collect_trial_data
from protocol.reply_parser import parse_ack
from transport.serial_interface import SerialLineIO
from ui.graphing import RuntimeMonitor, prompt_launch_gui, run_graph_tool


class EarlyStopOptimization(RuntimeError):
    pass


AXIS_NAMES = ("kp", "ki", "kd")
SURROGATE_FEATURE_NAMES = ("kp", "ki", "kd", "desired_output", "frequency_khz")
BayesSpaceDim = Any


# Print log lines with a simple HH:MM:SS timestamp.
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# Show a minimal startup menu and return the selected action.
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


# Ask for the target power value used to score each PID candidate.
def prompt_goal_power_output(default_value: float) -> float:
    while True:
        raw = input(f"Enter goal power output [{default_value}]: ").strip()
        if raw == "":
            return float(default_value)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a numeric value.", flush=True)


# Ask for how many PID trials to run. Each trial executes 5 laser tests.
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
    except Exception as e:
        log(f"Warning: Could not read current program values: {e}. Sending requested program values directly.")

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
    except Exception as e:
        log(f"Warning: Could not load startup defaults from hardware: {e}")
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


# Run one PID candidate across repeated timed tests.
def run_trial(
    io: SerialLineIO,
    kp: float,
    ki: float,
    kd: float,
    desired_output: float,
    apply_pid_update: bool = True,
    repeats: int = 5,
    test_duration_s: float = 12.0,
    startup_grace_s: float = 2.0,
    settled_window_samples: int = 5,
    duration: float = 8.0,
    kp_max: float = 1.0,
    ki_max: float = 1.0,
    kd_max: float = 0.2,
    monitor: RuntimeMonitor | None = None,
    trial_index: int | None = None,
    phase_name: str | None = None,
    phase_trial_index: int | None = None,
    phase_trial_total: int | None = None,
    overall_trial_index: int | None = None,
    repeat_cancel_osc_threshold: float = 0.35,
    repeat_cancel_score_regression_pct: float = 8.0,
):
    """Run one PID candidate through repeated laser tests and collect telemetry.

    A "trial" in this project means one PID tuple (kp, ki, kd) tested several
    times (`repeats`) so scoring is less sensitive to one noisy run.
    """
    _ = duration  # Kept for CLI compatibility with older call sites.
    log(f"Starting trial: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

    # Clamp gains to safe search limits before sending anything to hardware.
    kp = float(np.clip(kp, 0.0, kp_max))
    ki = float(np.clip(ki, 0.0, ki_max))
    kd = float(np.clip(kd, 0.0, kd_max))

    # Enable the debug stream so B0 power telemetry is available during tests.
    try:
        io.write_command_expect_ok_ack("000B", command_id_hex2=CMD.SET_DEBUG, timeout=2.0)
        log("SET_DEBUG acknowledged with *00")
    except Exception as e:
        raise RuntimeError(f"SET_DEBUG did not receive success ACK '*00': {e}") from e

    # Read current PID values from the laser once at trial start.
    try:
        current_pid = io.get_pid_values(timeout=2.0)
        log(
            "Current PID values: "
            f"PW Kp={current_pid['pw_kp']:.4f}, "
            f"Ki={current_pid['pw_ki']:.4f}, "
            f"Kd={current_pid['pw_kd']:.4f}"
        )
    except Exception as e:
        log(f"Warning: Could not read current PID values: {e}. Using defaults for PP parameters.")
        current_pid = None

    # Convert sample interval to seconds. Many controllers report milliseconds here.
    sample_interval_s = None
    if current_pid is not None:
        raw_si = float(current_pid.get("sample_interval", 0.0))
        if raw_si > 0:
            sample_interval_s = raw_si / 1000.0 if raw_si > 1.0 else raw_si
            log(f"Using telemetry sample interval: {sample_interval_s:.6f}s")

    # First trial can run with the laser's live PID values as a baseline.
    if apply_pid_update:
        ack = io.set_pid_values(
            pw_kp=kp,
            pw_ki=ki,
            pw_kd=kd,
            current_values=current_pid,
            timeout=2.0,
        )
        ok_ack, _ = parse_ack(ack)
        if not ack.startswith("*"):
            log(f"Warning: Unexpected SET_PID acknowledgment: {ack}")
        elif not ok_ack:
            log(f"Warning: SET_PID returned error code: {ack}")
    else:
        log("First trial: using current laser PID values without override")

    display_kp, display_ki, display_kd = kp, ki, kd
    if (not apply_pid_update) and current_pid is not None:
        display_kp = float(current_pid["pw_kp"])
        display_ki = float(current_pid["pw_ki"])
        display_kd = float(current_pid["pw_kd"])
    if monitor is not None:
        monitor.set_target(desired_output)
        monitor.set_pid_values(display_kp, display_ki, display_kd)
        if phase_name is not None and phase_trial_index is not None:
            progress = f"{phase_name} trial {phase_trial_index}"
            if phase_trial_total is not None:
                progress = f"{progress}/{phase_trial_total}"
            if overall_trial_index is not None:
                progress = f"{progress} | overall {overall_trial_index}"
            monitor.set_progress(f"{progress} | configuring hardware")
        elif trial_index is not None:
            monitor.set_progress(f"Trial {trial_index} | configuring hardware")

    # Prepare output arrays for all repeats under this PID candidate.
    t_vals, y_vals, u_vals, status_vals = [], [], [], []
    per_test_powers: list[np.ndarray] = []
    per_test_times: list[np.ndarray] = []
    per_test_meta: list[dict] = []
    repeat_scores: list[float] = []
    cancelled_candidate = False
    cancel_reason = ""

    # Put the laser into run mode and open shutter before individual test starts.
    io.write_command_expect_ok_ack("", command_id_hex2=CMD.RUN, timeout=2.0)
    io.write_command_expect_ok_ack("1", command_id_hex2=CMD.SHUTTER_CONTROL, timeout=2.0)

    try:
        # Repeat the same PID candidate several times to reduce one-off noise.
        for rep in range(repeats):
            log(f"Test {rep + 1}/{repeats}: START")
            if monitor is not None:
                monitor.begin_test(
                    phase_name=(phase_name or "Trial"),
                    phase_trial_index=(phase_trial_index or trial_index or 0),
                    phase_trial_total=phase_trial_total,
                    repeat_index=rep + 1,
                    repeats=repeats,
                    overall_trial_index=overall_trial_index,
                )
            io.write_command_expect_ok_ack(
                "",
                command_id_hex2=CMD.START,
                timeout=2.0,
                # Some controller firmware returns *08 here when START is
                # accepted from the current machine state.
                accepted_codes=("00", "08", "80"),
            )

            test_meta = {
                "invalid": False,
                "reason": "",
                "settled": False,
                "strict_bad_rate": 1.0,
                "oscillation_rate": 1.0,
                "stopped_early": False,
                "start_skewed": False,
                "start_skew_error": 0.0,
            }
            recent_within_5pct = []
            strict_bad_count = 0
            strict_total = 0
            settled_errors: list[float] = []
            first_seen = False

            base = max(abs(desired_output), 1e-6)
            limit_30 = 0.30 * base
            limit_5 = 0.05 * base
            limit_1 = 0.01 * base
            osc_deadband = 0.03 * base

            def on_sample(t_val, mapped) -> bool:
                nonlocal strict_bad_count, strict_total, first_seen
                y_val = float(mapped["process_value"])
                if monitor is not None:
                    monitor.append_sample(t_val, y_val, status=str(mapped.get("status", "RUNNING")))
                err = y_val - desired_output
                abs_err = abs(err)

                if not first_seen:
                    first_seen = True
                    # Use the B0 initial power field (first section) when available.
                    first_power = float(mapped.get("initial_power", y_val))
                    first_err = abs(first_power - desired_output)
                    low_limit = desired_output - limit_30
                    high_limit = desired_output + limit_30
                    log(
                        f"Initial power check -> value={first_power:.4f}, "
                        f"allowed=[{low_limit:.4f}, {high_limit:.4f}]"
                    )
                    if first_err > limit_30:
                        test_meta["start_skewed"] = True
                        test_meta["start_skew_error"] = float(first_err)
                        test_meta["reason"] = (
                            f"start skewed beyond +/-30% "
                            f"(initial={first_power:.4f}, target={desired_output:.4f})"
                        )
                        log(f"Test {rep + 1}/{repeats} note: {test_meta['reason']}")

                if t_val < startup_grace_s:
                    return False

                within_5 = abs_err <= limit_5
                recent_within_5pct.append(within_5)
                if len(recent_within_5pct) > settled_window_samples:
                    recent_within_5pct.pop(0)

                if (not test_meta["settled"]) and len(recent_within_5pct) == settled_window_samples:
                    if all(recent_within_5pct):
                        test_meta["settled"] = True

                if test_meta["settled"]:
                    if abs_err > limit_5:
                        test_meta["invalid"] = True
                        test_meta["reason"] = (
                            f"settled reading out of +/-5% "
                            f"(value={y_val:.4f}, target={desired_output:.4f})"
                        )
                        try:
                            io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
                            test_meta["stopped_early"] = True
                        except Exception as stop_err:
                            log(f"Warning: failed to send immediate STOP on invalid test: {stop_err}")
                        return True

                    strict_total += 1
                    if abs_err > limit_1:
                        strict_bad_count += 1

                    settled_errors.append(float(err))

                return False

            # Collect telemetry for a fixed window, then stop this test pass.
            rt, ry, ru, rs = collect_trial_data(
                io,
                line_timeout=0.5,
                sample_interval_s=sample_interval_s,
                duration_s=test_duration_s,
                stop_on_done=False,
                on_sample=on_sample,
            )

            # Shift each repeat's time axis so combined arrays stay monotonic.
            t_offset = rep * test_duration_s
            t_vals.extend([float(v) + t_offset for v in rt])
            y_vals.extend(ry)
            u_vals.extend(ru)
            status_vals.extend(rs)
            per_test_powers.append(np.array(ry, dtype=float))
            per_test_times.append(np.array(rt, dtype=float))

            if strict_total > 0:
                test_meta["strict_bad_rate"] = strict_bad_count / strict_total
            else:
                test_meta["strict_bad_rate"] = 1.0

            significant_error_signs = []
            for settled_err in settled_errors:
                if abs(settled_err) < osc_deadband:
                    continue
                significant_error_signs.append(1 if settled_err > 0 else -1)
            if len(significant_error_signs) >= 2:
                sign_flips = sum(
                    1
                    for prev_sign, curr_sign in zip(significant_error_signs, significant_error_signs[1:])
                    if prev_sign != curr_sign
                )
                test_meta["oscillation_rate"] = sign_flips / float(len(significant_error_signs) - 1)
            else:
                test_meta["oscillation_rate"] = 0.0 if test_meta["settled"] else 1.0

            if not ry and not test_meta["invalid"]:
                test_meta["invalid"] = True
                test_meta["reason"] = "no samples collected"
            if not test_meta["settled"] and not test_meta["invalid"]:
                # Did not settle within test window: keep valid, but penalize in metrics.
                test_meta["reason"] = "did not settle"

            per_test_meta.append(test_meta)
            repeat_score = score_single_repeat(per_test_powers[-1], test_meta, desired_output)
            repeat_scores.append(repeat_score)

            # Print quick power stats for this repeat to spot unstable behavior.
            if ry:
                avg_power = float(np.mean(ry))
                min_power = float(np.min(ry))
                max_power = float(np.max(ry))
                log(
                    f"Test {rep + 1}/{repeats} current_power -> "
                    f"avg={avg_power:.4f}, min={min_power:.4f}, max={max_power:.4f}, n={len(ry)}"
                )
            else:
                log(f"Test {rep + 1}/{repeats} current_power -> no samples")

            if test_meta["invalid"]:
                log(f"Test {rep + 1}/{repeats} invalid: {test_meta['reason']}")
            elif test_meta["reason"]:
                log(f"Test {rep + 1}/{repeats} note: {test_meta['reason']}")

            if rep >= 2:
                prev_repeat_score = repeat_scores[-2]
                score_regression_pct = 0.0
                if prev_repeat_score > 1e-9:
                    score_regression_pct = 100.0 * (repeat_score - prev_repeat_score) / prev_repeat_score
                if test_meta["oscillation_rate"] >= repeat_cancel_osc_threshold:
                    cancelled_candidate = True
                    cancel_reason = (
                        f"repeat {rep + 1} oscillation too high "
                        f"({test_meta['oscillation_rate']:.3f} >= {repeat_cancel_osc_threshold:.3f})"
                    )
                elif score_regression_pct >= float(repeat_cancel_score_regression_pct):
                    cancelled_candidate = True
                    cancel_reason = (
                        f"repeat {rep + 1} score regressed by {score_regression_pct:.1f}% "
                        f"({repeat_score:.3f} > {prev_repeat_score:.3f})"
                    )

            if not test_meta["stopped_early"]:
                io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
                log(f"Test {rep + 1}/{repeats}: STOP")
                if monitor is not None:
                    if phase_name is not None and phase_trial_index is not None:
                        progress = f"{phase_name} trial {phase_trial_index}"
                        if phase_trial_total is not None:
                            progress = f"{progress}/{phase_trial_total}"
                        if overall_trial_index is not None:
                            progress = f"{progress} | overall {overall_trial_index}"
                        monitor.set_progress(f"{progress} | test {rep + 1}/{repeats} stopped")
                    elif trial_index is not None:
                        monitor.set_progress(f"Trial {trial_index} | test {rep + 1}/{repeats} stopped")
            else:
                log(f"Test {rep + 1}/{repeats}: STOP (already sent on invalid condition)")

            if cancelled_candidate:
                log(f"Cancelling remaining repeats for this PID candidate: {cancel_reason}")
                break
    finally:
        # Always leave hardware in a safe idle state at end of a candidate.
        try:
            io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
        except Exception:
            pass
        io.write_command_expect_ok_ack("0", command_id_hex2=CMD.SHUTTER_CONTROL, timeout=2.0)
        io.write_command_expect_ok_ack("", command_id_hex2=CMD.STANDBY, timeout=2.0)
        log("End of PID set: shutter closed, standby set")
        if monitor is not None:
            if phase_name is not None and phase_trial_index is not None:
                progress = f"{phase_name} trial {phase_trial_index}"
                if phase_trial_total is not None:
                    progress = f"{progress}/{phase_trial_total}"
                if overall_trial_index is not None:
                    progress = f"{progress} | overall {overall_trial_index}"
                monitor.set_progress(f"{progress} | shutter closed, standby set")
            elif trial_index is not None:
                monitor.set_progress(f"Trial {trial_index} | shutter closed, standby set")

    aborted = any(s == "ABORT" for s in status_vals)
    start_skew_count = sum(1 for meta in per_test_meta if bool(meta.get("start_skewed", False)))
    start_skew_threshold = max(2, (len(per_test_meta) // 2) + 1) if per_test_meta else 2
    if start_skew_count >= start_skew_threshold:
        summary_reason = (
            f"start skew exceeded +/-30% in {start_skew_count}/{len(per_test_meta)} repeats"
        )
        for meta in per_test_meta:
            if bool(meta.get("start_skewed", False)) and not bool(meta.get("invalid", False)):
                meta["invalid"] = True
                meta["reason"] = summary_reason
        if repeat_scores:
            repeat_scores = [
                999.0 if bool(meta.get("start_skewed", False)) else score
                for meta, score in zip(per_test_meta, repeat_scores)
            ]
        log(f"Candidate start-skew rule triggered: {summary_reason}")
    if aborted:
        log("Warning: Trial aborted due to safety condition")
    if cancelled_candidate:
        log(f"Trial ended early after repeated-test regression: {cancel_reason}")

    return (
        np.array(t_vals),
        np.array(y_vals),
        np.array(u_vals),
        aborted,
        current_pid,
        per_test_powers,
        per_test_times,
        per_test_meta,
        cancelled_candidate,
        cancel_reason,
    )


def _safe_mean(values: list[float], default: float) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float(default)


def _safe_std(values: list[float], default: float) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(finite)) if finite else float(default)


def _clean_trace(times: np.ndarray, readings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times_arr = np.asarray(times, dtype=float).flatten()
    readings_arr = np.asarray(readings, dtype=float).flatten()
    n = int(min(times_arr.size, readings_arr.size))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    times_arr = times_arr[:n]
    readings_arr = readings_arr[:n]
    mask = np.isfinite(times_arr) & np.isfinite(readings_arr)
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)
    times_arr = times_arr[mask]
    readings_arr = readings_arr[mask]
    if times_arr.size <= 1:
        return times_arr, readings_arr
    order = np.argsort(times_arr, kind="stable")
    return times_arr[order], readings_arr[order]


def _sample_widths(times: np.ndarray) -> np.ndarray:
    if times.size <= 1:
        return np.ones(int(max(1, times.size)), dtype=float)
    deltas = np.diff(times)
    positive = deltas[deltas > 1e-9]
    default_dt = float(np.median(positive)) if positive.size else 1.0
    widths = np.empty(times.size, dtype=float)
    widths[:-1] = np.where(deltas > 1e-9, deltas, default_dt)
    widths[-1] = default_dt
    return widths


def _integral(times: np.ndarray, values: np.ndarray) -> float:
    if times.size == 0 or values.size == 0:
        return 0.0
    if times.size == 1:
        return float(abs(values[0]))
    deltas = np.diff(times)
    if deltas.size == 0:
        return float(abs(values[0]))
    left = np.asarray(values[:-1], dtype=float)
    right = np.asarray(values[1:], dtype=float)
    area = 0.5 * (left + right) * deltas
    return float(np.sum(area))


def _first_index(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    return int(idx[0])


def _compute_rise_time(times: np.ndarray, readings: np.ndarray, target: float) -> float | None:
    if times.size < 2 or readings.size < 2:
        return None
    start = float(readings[0])
    amplitude = float(target - start)
    if abs(amplitude) < 1e-9:
        return 0.0 if abs(start - target) < max(abs(target) * 0.05, 1.0) else None
    norm = (readings - start) / amplitude
    i10 = _first_index(norm >= 0.1)
    i90 = _first_index(norm >= 0.9)
    if i10 is None or i90 is None or i90 < i10:
        return None
    return float(times[i90] - times[i10])


def _compute_settling_time(
    times: np.ndarray,
    abs_error: np.ndarray,
    tolerance: float,
    *,
    settled_window_samples: int = 5,
    settle_success_ratio: float = 0.85,
) -> float | None:
    if times.size == 0 or abs_error.size == 0:
        return None
    if tolerance <= 0:
        return None
    within = abs_error <= tolerance
    window = max(2, int(min(settled_window_samples, max(2, times.size))))
    for start_idx in range(0, max(1, times.size - window + 1)):
        window_ok = within[start_idx : start_idx + window]
        if window_ok.size < window or not np.all(window_ok):
            continue
        tail = within[start_idx:]
        if float(np.mean(tail.astype(float))) >= float(settle_success_ratio):
            return float(times[start_idx])
    return None


def _count_sign_changes(values: np.ndarray, deadband: float) -> int:
    signs = []
    for value in values:
        if abs(float(value)) <= deadband:
            continue
        signs.append(1 if value > 0 else -1)
    if len(signs) < 2:
        return 0
    return int(sum(1 for prev, curr in zip(signs, signs[1:]) if prev != curr))


def extract_early_trace_features(
    times: np.ndarray,
    readings: np.ndarray,
    desired_output: float,
    *,
    early_window_s: float = 2.0,
) -> dict:
    """Summarise the first part of a trace for future early-abort models."""
    times_arr, readings_arr = _clean_trace(times, readings)
    if times_arr.size == 0:
        return {
            "early_window_s": float(early_window_s),
            "early_sample_count": 0.0,
            "early_slope": 0.0,
            "early_mean_error": float(abs(desired_output)),
            "early_max_error": float(abs(desired_output)),
        }

    window_mask = times_arr <= (float(times_arr[0]) + float(max(early_window_s, 0.1)))
    early_times = times_arr[window_mask]
    early_readings = readings_arr[window_mask]
    if early_times.size < 2:
        early_times = times_arr[: min(3, times_arr.size)]
        early_readings = readings_arr[: early_times.size]
    early_errors = early_readings - float(desired_output)
    duration = float(max(early_times[-1] - early_times[0], 1e-6))
    early_slope = float((early_readings[-1] - early_readings[0]) / duration) if early_times.size >= 2 else 0.0
    return {
        "early_window_s": float(early_window_s),
        "early_sample_count": float(early_times.size),
        "early_slope": early_slope,
        "early_mean_error": float(np.mean(np.abs(early_errors))) if early_errors.size else 0.0,
        "early_max_error": float(np.max(np.abs(early_errors))) if early_errors.size else 0.0,
    }


def extract_repeat_features(
    times: np.ndarray,
    readings: np.ndarray,
    desired_output: float,
    *,
    settled_window_samples: int = 5,
    tolerance_pct: float = 0.05,
    hold_tail_fraction: float = 0.33,
    hold_min_samples: int = 5,
) -> dict:
    """Compute lightweight control features from one repeat trace."""
    times_arr, readings_arr = _clean_trace(times, readings)
    empty_result = {
        "start_error": 999.0,
        "track_error": 999.0,
        "deviation": 999.0,
        "max_error": 999.0,
        "overshoot_pct": 100.0,
        "settling_time_s": math.nan,
        "rise_time_s": math.nan,
        "steady_state_error": 999.0,
        "iae": 999.0,
        "ise": 999.0,
        "itae": 999.0,
        "peak_value": math.nan,
        "peak_time_s": math.nan,
        "time_in_tolerance_s": 0.0,
        "time_to_first_tolerance_s": math.nan,
        "post_settle_variance": 999.0,
        "early_slope": 0.0,
        "oscillation_count": 0.0,
        "area_above_target": 0.0,
        "area_below_target": 0.0,
        "trace_duration_s": 0.0,
        "hold_duration_s": 0.0,
        "hold_mean_error": 999.0,
        "hold_variance": 999.0,
        "hold_drift": 999.0,
        "hold_time_in_tolerance_ratio": 0.0,
        "hold_oscillation_count": 0.0,
        "hold_quality": 999.0,
    }
    if times_arr.size == 0:
        return empty_result

    target = float(desired_output)
    base = max(abs(target), 1e-6)
    tolerance = max(base * float(tolerance_pct), 1e-6)
    widths = _sample_widths(times_arr)
    error = readings_arr - target
    abs_error = np.abs(error)
    start_error = abs(float(readings_arr[0]) - target)
    track_error = float(np.mean(abs_error))
    deviation = float(np.std(readings_arr))
    max_error = float(np.max(abs_error))
    peak_idx = int(np.argmax(readings_arr))
    peak_value = float(readings_arr[peak_idx])
    peak_time_s = float(times_arr[peak_idx])
    overshoot_pct = max(0.0, (peak_value - target) / base * 100.0)
    settling_time_s = _compute_settling_time(
        times_arr,
        abs_error,
        tolerance,
        settled_window_samples=settled_window_samples,
    )
    rise_time_s = _compute_rise_time(times_arr, readings_arr, target)
    steady_slice = readings_arr[-max(3, readings_arr.size // 4) :]
    steady_state_error = float(np.mean(steady_slice) - target) if steady_slice.size else 999.0
    iae = _integral(times_arr, abs_error)
    ise = _integral(times_arr, error**2)
    itae = _integral(times_arr, np.abs(error) * np.maximum(times_arr - float(times_arr[0]), 0.0))
    within_tol = abs_error <= tolerance
    time_in_tolerance_s = float(np.sum(widths[within_tol])) if within_tol.size else 0.0
    first_tol_idx = _first_index(within_tol)
    time_to_first_tolerance_s = float(times_arr[first_tol_idx]) if first_tol_idx is not None else math.nan
    post_settle_variance = deviation
    if settling_time_s is not None:
        post_mask = times_arr >= settling_time_s
        if np.any(post_mask):
            post_settle_variance = float(np.var(readings_arr[post_mask]))
    early_features = extract_early_trace_features(times_arr, readings_arr, target)
    area_above_target = _integral(times_arr, np.maximum(error, 0.0))
    area_below_target = _integral(times_arr, np.maximum(-error, 0.0))
    oscillation_count = _count_sign_changes(error, deadband=0.03 * base)
    trace_duration_s = float(max(times_arr[-1] - times_arr[0], 0.0)) if times_arr.size >= 2 else 0.0

    hold_sample_count = min(
        readings_arr.size,
        max(int(math.ceil(readings_arr.size * float(hold_tail_fraction))), int(hold_min_samples)),
    )
    hold_readings = readings_arr[-hold_sample_count:]
    hold_times = times_arr[-hold_sample_count:]
    hold_errors = hold_readings - target
    hold_widths = _sample_widths(hold_times)
    hold_duration_s = float(np.sum(hold_widths)) if hold_widths.size else 0.0
    hold_mean_error = float(np.mean(hold_readings) - target) if hold_readings.size else 999.0
    hold_variance = float(np.var(hold_readings)) if hold_readings.size else 999.0
    hold_midpoint = max(1, hold_readings.size // 2)
    hold_first = hold_readings[:hold_midpoint]
    hold_second = hold_readings[hold_midpoint:]
    if hold_first.size and hold_second.size:
        hold_drift = float(abs(np.mean(hold_second) - np.mean(hold_first)))
    else:
        hold_drift = float(abs(hold_mean_error))
    hold_within_tol = np.abs(hold_errors) <= tolerance
    if hold_widths.size and np.sum(hold_widths) > 0:
        hold_time_in_tolerance_ratio = float(np.sum(hold_widths[hold_within_tol]) / np.sum(hold_widths))
    elif hold_within_tol.size:
        hold_time_in_tolerance_ratio = float(np.mean(hold_within_tol))
    else:
        hold_time_in_tolerance_ratio = 0.0
    hold_oscillation_count = float(_count_sign_changes(hold_errors, deadband=0.02 * base))
    hold_mean_error_pct = 100.0 * abs(hold_mean_error) / base
    hold_std_pct = 100.0 * math.sqrt(max(hold_variance, 0.0)) / base
    hold_drift_pct = 100.0 * hold_drift / base
    hold_tolerance_miss_pct = 100.0 * max(0.0, 1.0 - hold_time_in_tolerance_ratio)
    hold_quality = (
        3.0 * hold_mean_error_pct
        + 2.0 * hold_tolerance_miss_pct
        + 2.0 * hold_std_pct
        + 1.5 * hold_drift_pct
        + 4.0 * hold_oscillation_count
    )
    return {
        "start_error": start_error,
        "track_error": track_error,
        "deviation": deviation,
        "max_error": max_error,
        "overshoot_pct": float(overshoot_pct),
        "settling_time_s": float(settling_time_s) if settling_time_s is not None else math.nan,
        "rise_time_s": float(rise_time_s) if rise_time_s is not None else math.nan,
        "steady_state_error": steady_state_error,
        "iae": float(iae),
        "ise": float(ise),
        "itae": float(itae),
        "peak_value": peak_value,
        "peak_time_s": peak_time_s,
        "time_in_tolerance_s": time_in_tolerance_s,
        "time_to_first_tolerance_s": time_to_first_tolerance_s,
        "post_settle_variance": float(post_settle_variance),
        "early_slope": float(early_features["early_slope"]),
        "oscillation_count": float(oscillation_count),
        "area_above_target": float(area_above_target),
        "area_below_target": float(area_below_target),
        "early_mean_error": float(early_features["early_mean_error"]),
        "early_max_error": float(early_features["early_max_error"]),
        "trace_duration_s": float(trace_duration_s),
        "hold_duration_s": float(hold_duration_s),
        "hold_mean_error": float(hold_mean_error),
        "hold_variance": float(hold_variance),
        "hold_drift": float(hold_drift),
        "hold_time_in_tolerance_ratio": float(hold_time_in_tolerance_ratio),
        "hold_oscillation_count": float(hold_oscillation_count),
        "hold_quality": float(hold_quality),
    }


def aggregate_repeat_features(repeat_features: list[dict]) -> dict:
    """Aggregate repeat-level trace features into candidate-level features."""
    if not repeat_features:
        return {
            "feature_early_slope_mean": 0.0,
            "feature_peak_value_mean": math.nan,
            "feature_peak_time_s_mean": math.nan,
            "feature_overshoot_pct_mean": 100.0,
            "feature_time_to_first_tolerance_s_mean": math.nan,
            "feature_time_in_tolerance_s_mean": 0.0,
            "feature_oscillation_count_mean": 0.0,
            "feature_area_above_target_mean": 0.0,
            "feature_area_below_target_mean": 0.0,
            "feature_post_settle_variance_mean": 999.0,
            "feature_early_mean_error_mean": 999.0,
            "feature_early_max_error_mean": 999.0,
            "feature_hold_duration_s_mean": 0.0,
            "feature_hold_mean_error_mean": 999.0,
            "feature_hold_variance_mean": 999.0,
            "feature_hold_drift_mean": 999.0,
            "feature_hold_time_in_tolerance_ratio_mean": 0.0,
            "feature_hold_oscillation_count_mean": 0.0,
            "feature_hold_quality_mean": 999.0,
            "feature_peak_value_std": 0.0,
            "feature_overshoot_pct_std": 0.0,
        }

    def values(key: str) -> list[float]:
        return [float(item[key]) for item in repeat_features if key in item and np.isfinite(float(item[key]))]

    return {
        "feature_early_slope_mean": _safe_mean(values("early_slope"), 0.0),
        "feature_peak_value_mean": _safe_mean(values("peak_value"), math.nan),
        "feature_peak_time_s_mean": _safe_mean(values("peak_time_s"), math.nan),
        "feature_overshoot_pct_mean": _safe_mean(values("overshoot_pct"), 100.0),
        "feature_time_to_first_tolerance_s_mean": _safe_mean(values("time_to_first_tolerance_s"), math.nan),
        "feature_time_in_tolerance_s_mean": _safe_mean(values("time_in_tolerance_s"), 0.0),
        "feature_oscillation_count_mean": _safe_mean(values("oscillation_count"), 0.0),
        "feature_area_above_target_mean": _safe_mean(values("area_above_target"), 0.0),
        "feature_area_below_target_mean": _safe_mean(values("area_below_target"), 0.0),
        "feature_post_settle_variance_mean": _safe_mean(values("post_settle_variance"), 999.0),
        "feature_early_mean_error_mean": _safe_mean(values("early_mean_error"), 999.0),
        "feature_early_max_error_mean": _safe_mean(values("early_max_error"), 999.0),
        "feature_hold_duration_s_mean": _safe_mean(values("hold_duration_s"), 0.0),
        "feature_hold_mean_error_mean": _safe_mean(values("hold_mean_error"), 999.0),
        "feature_hold_variance_mean": _safe_mean(values("hold_variance"), 999.0),
        "feature_hold_drift_mean": _safe_mean(values("hold_drift"), 999.0),
        "feature_hold_time_in_tolerance_ratio_mean": _safe_mean(values("hold_time_in_tolerance_ratio"), 0.0),
        "feature_hold_oscillation_count_mean": _safe_mean(values("hold_oscillation_count"), 0.0),
        "feature_hold_quality_mean": _safe_mean(values("hold_quality"), 999.0),
        "feature_peak_value_std": _safe_std(values("peak_value"), 0.0),
        "feature_overshoot_pct_std": _safe_std(values("overshoot_pct"), 0.0),
    }


def compute_trial_metrics(
    per_test_powers: list[np.ndarray],
    per_test_times: list[np.ndarray],
    per_test_meta: list[dict],
    desired_output: float,
    *,
    settled_window_samples: int = 5,
):
    """Convert repeated traces into aggregate control metrics and features."""
    repeat_features: list[dict] = []
    strict_bad_rates = []
    oscillation_rates = []
    invalid_flags = []
    per_test_scores_unweighted = []

    for readings, times, meta in zip(per_test_powers, per_test_times, per_test_meta):
        features = extract_repeat_features(
            times,
            readings,
            desired_output,
            settled_window_samples=settled_window_samples,
        )
        repeat_features.append(features)
        strict_bad_rates.append(float(meta.get("strict_bad_rate", 1.0)))
        oscillation_rates.append(float(meta.get("oscillation_rate", 1.0)))
        invalid_flags.append(1.0 if bool(meta.get("invalid", False)) else 0.0)
        per_test_scores_unweighted.append(
            features["track_error"]
            + features["max_error"]
            + features["overshoot_pct"]
            + abs(features["steady_state_error"])
            + features["iae"]
            + strict_bad_rates[-1]
            + oscillation_rates[-1]
        )

    metrics = {
        "start_error": _safe_mean([f["start_error"] for f in repeat_features], 999.0),
        "track_error": _safe_mean([f["track_error"] for f in repeat_features], 999.0),
        "deviation": _safe_mean([f["deviation"] for f in repeat_features], 999.0),
        "max_error": _safe_mean([f["max_error"] for f in repeat_features], 999.0),
        "overshoot_pct": _safe_mean([f["overshoot_pct"] for f in repeat_features], 100.0),
        "settling_time_s": _safe_mean([f["settling_time_s"] for f in repeat_features], 999.0),
        "rise_time_s": _safe_mean([f["rise_time_s"] for f in repeat_features], 999.0),
        "steady_state_error": _safe_mean([f["steady_state_error"] for f in repeat_features], 999.0),
        "iae": _safe_mean([f["iae"] for f in repeat_features], 999.0),
        "ise": _safe_mean([f["ise"] for f in repeat_features], 999.0),
        "itae": _safe_mean([f["itae"] for f in repeat_features], 999.0),
        "peak_value": _safe_mean([f["peak_value"] for f in repeat_features], math.nan),
        "peak_time_s": _safe_mean([f["peak_time_s"] for f in repeat_features], 999.0),
        "time_in_tolerance_s": _safe_mean([f["time_in_tolerance_s"] for f in repeat_features], 0.0),
        "time_to_first_tolerance_s": _safe_mean([f["time_to_first_tolerance_s"] for f in repeat_features], 999.0),
        "post_settle_variance": _safe_mean([f["post_settle_variance"] for f in repeat_features], 999.0),
        "trace_duration_s": _safe_mean([f["trace_duration_s"] for f in repeat_features], 0.0),
        "hold_duration_s": _safe_mean([f["hold_duration_s"] for f in repeat_features], 0.0),
        "hold_mean_error": _safe_mean([f["hold_mean_error"] for f in repeat_features], 999.0),
        "hold_variance": _safe_mean([f["hold_variance"] for f in repeat_features], 999.0),
        "hold_drift": _safe_mean([f["hold_drift"] for f in repeat_features], 999.0),
        "hold_time_in_tolerance_ratio": _safe_mean([f["hold_time_in_tolerance_ratio"] for f in repeat_features], 0.0),
        "hold_oscillation_count": _safe_mean([f["hold_oscillation_count"] for f in repeat_features], 0.0),
        "hold_quality": _safe_mean([f["hold_quality"] for f in repeat_features], 999.0),
        "strict_bad_rate": _safe_mean(strict_bad_rates, 1.0),
        "oscillation_rate": _safe_mean(oscillation_rates, 1.0),
        "invalid_ratio": _safe_mean(invalid_flags, 1.0),
        "repeatability": _safe_std(per_test_scores_unweighted, 999.0),
    }
    metrics.update(aggregate_repeat_features(repeat_features))
    return metrics, repeat_features


def score_single_repeat(readings: np.ndarray, meta: dict, desired_output: float) -> float:
    """Score one repeat quickly so unstable candidates can be stopped early."""
    if readings.size == 0:
        return 999.0

    start_power = float(readings[0])
    abs_error = np.abs(readings - desired_output)
    start_error = abs(start_power - desired_output)
    track_error = float(np.mean(abs_error))
    deviation = float(np.std(readings))
    max_error = float(np.max(abs_error))
    strict_bad_rate = float(meta.get("strict_bad_rate", 1.0))
    oscillation_rate = float(meta.get("oscillation_rate", 1.0))
    hold_error = float(np.mean(readings[-max(5, readings.size // 3) :]) - desired_output)
    hold_variance = float(np.var(readings[-max(5, readings.size // 3) :]))
    return (
        0.25 * start_error
        + 0.50 * track_error
        + 0.75 * deviation
        + 0.75 * max_error
        + 2.0 * abs(hold_error)
        + 1.5 * hold_variance
        + 3.0 * strict_bad_rate
        + 4.0 * oscillation_rate
    )


class OnlineSurrogateModel:
    """Small refit-on-update surrogate wrapper for embedded-friendly tuning."""

    def __init__(self, model_name: str, *, random_state: int = 42):
        self.model_name = str(model_name)
        self.random_state = int(random_state)
        self.model = None
        self.is_available = False
        self.last_error = ""
        self.last_fit_count = 0

    def _build_model(self):
        if not HAVE_SKLEARN or self.model_name == "none":
            raise RuntimeError("scikit-learn unavailable")
        if self.model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=48,
                max_depth=10,
                min_samples_leaf=2,
                random_state=self.random_state,
            )
        return ExtraTreesRegressor(
            n_estimators=48,
            max_depth=10,
            min_samples_leaf=2,
            random_state=self.random_state,
        )

    def fit(self, rows: list[dict], *, min_samples: int) -> bool:
        usable = [row for row in rows if np.isfinite(float(row.get("score", math.inf)))]
        if len(usable) < int(min_samples):
            self.is_available = False
            self.model = None
            self.last_fit_count = len(usable)
            return False
        try:
            x = np.asarray(
                [[float(row[name]) for name in SURROGATE_FEATURE_NAMES] for row in usable],
                dtype=float,
            )
            y = np.asarray([float(row["score"]) for row in usable], dtype=float)
            self.model = self._build_model()
            self.model.fit(x, y)
            self.is_available = True
            self.last_error = ""
            self.last_fit_count = len(usable)
            return True
        except Exception as exc:
            self.model = None
            self.is_available = False
            self.last_error = str(exc)
            self.last_fit_count = len(usable)
            return False

    def predict(self, candidates: list[tuple[float, float, float]], *, desired_output: float, frequency_khz: int) -> list[float]:
        if not self.is_available or self.model is None or not candidates:
            return [math.nan for _ in candidates]
        x = np.asarray(
            [
                [float(kp), float(ki), float(kd), float(desired_output), float(frequency_khz)]
                for kp, ki, kd in candidates
            ],
            dtype=float,
        )
        try:
            preds = self.model.predict(x)
            return [float(v) for v in preds]
        except Exception as exc:
            self.is_available = False
            self.last_error = str(exc)
            return [math.nan for _ in candidates]


def build_surrogate_training_row(
    kp: float,
    ki: float,
    kd: float,
    *,
    desired_output: float,
    frequency_khz: int,
    score: float,
) -> dict:
    return {
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "desired_output": float(desired_output),
        "frequency_khz": float(frequency_khz),
        "score": float(score),
    }


def propose_surrogate_candidate(
    surrogate: OnlineSurrogateModel,
    *,
    best_pid: tuple[float, float, float] | None,
    safe_points: list[tuple[float, float, float]],
    observed_points: list[tuple[float, float, float]],
    desired_output: float,
    frequency_khz: int,
    rng: random.Random,
    pool_size: int,
    explore_prob: float,
    jitter_scale: float,
    step_kp: float,
    step_ki: float,
    step_kd: float,
    kp_max: float,
    ki_max: float,
    kd_max: float,
) -> tuple[tuple[float, float, float], float, str]:
    """Generate a bounded candidate pool and score it with the surrogate."""
    center = best_pid or (0.5 * kp_max, 0.5 * ki_max, 0.5 * kd_max)
    spans = np.asarray([max(step_kp, 1e-3), max(step_ki, 1e-3), max(step_kd, 1e-3)], dtype=float)
    candidates: list[tuple[float, float, float]] = []

    if safe_points:
        for point in safe_points[-min(6, len(safe_points)) :]:
            candidates.append(tuple(float(v) for v in point))

    for _ in range(max(8, int(pool_size))):
        if rng.random() < 0.5 and best_pid is not None:
            base = np.asarray(center, dtype=float)
        elif safe_points:
            base = np.asarray(rng.choice(safe_points), dtype=float)
        else:
            base = np.asarray(center, dtype=float)
        noise = np.asarray(
            [
                rng.uniform(-1.0, 1.0) * spans[0] * float(jitter_scale),
                rng.uniform(-1.0, 1.0) * spans[1] * float(jitter_scale),
                rng.uniform(-1.0, 1.0) * spans[2] * float(jitter_scale),
            ],
            dtype=float,
        )
        proposal = np.clip(base + noise, [0.0, 0.0, 0.0], [kp_max, ki_max, kd_max])
        candidates.append((float(proposal[0]), float(proposal[1]), float(proposal[2])))

    deduped: list[tuple[float, float, float]] = []
    seen = set()
    observed_rounded = {tuple(round(v, 6) for v in point) for point in observed_points}
    for candidate in candidates:
        rounded = tuple(round(v, 6) for v in candidate)
        if rounded in seen or rounded in observed_rounded:
            continue
        seen.add(rounded)
        deduped.append(candidate)

    if not deduped:
        deduped = [tuple(float(v) for v in center)]

    predictions = surrogate.predict(
        deduped,
        desired_output=desired_output,
        frequency_khz=frequency_khz,
    )
    ranked = sorted(zip(deduped, predictions), key=lambda item: item[1] if np.isfinite(item[1]) else float("inf"))
    if rng.random() < float(explore_prob) and len(ranked) > 1:
        choice = rng.choice(ranked[: min(4, len(ranked))] + ranked[-min(3, len(ranked)) :])
        return choice[0], float(choice[1]), "surrogate_explore"
    best_candidate, best_pred = ranked[0]
    return best_candidate, float(best_pred), "surrogate"


def score_controller(
    metrics: dict,
    *,
    w_start: float,
    w_track: float,
    w_dev: float,
    w_max: float,
    w_repeat: float,
    w_strict: float,
    w_osc: float,
    w_overshoot: float,
    w_settle: float,
    w_rise: float,
    w_steady: float,
    w_iae: float,
    w_ise: float,
    w_tolerance_time: float,
    w_post_var: float,
    w_hold: float,
    invalid_penalty: float,
    cancelled_candidate: bool,
    aborted: bool,
):
    """Combine control metrics into one scalar score (lower is better)."""
    hold_ratio = float(metrics.get("hold_time_in_tolerance_ratio", 0.0))
    track_discount = 0.25 if hold_ratio >= 0.80 else (0.50 if hold_ratio >= 0.50 else 1.0)
    score = (
        w_start * float(metrics["start_error"])
        + w_track * float(metrics["track_error"]) * track_discount
        + w_dev * float(metrics["deviation"])
        + w_max * float(metrics["max_error"])
        + w_repeat * float(metrics["repeatability"])
        + w_strict * float(metrics["strict_bad_rate"])
        + w_osc * float(metrics["oscillation_rate"])
        + w_overshoot * float(metrics["overshoot_pct"])
        + w_settle * float(metrics["settling_time_s"])
        + w_rise * float(metrics["rise_time_s"])
        + w_steady * abs(float(metrics["steady_state_error"]))
        + w_iae * float(metrics["iae"])
        + w_ise * float(metrics["ise"])
        + w_post_var * float(metrics["post_settle_variance"])
        + w_hold * float(metrics.get("hold_quality", 999.0))
        - w_tolerance_time * float(metrics["time_in_tolerance_s"])
        + invalid_penalty * float(metrics["invalid_ratio"])
    )
    if hold_ratio < 0.60:
        score += (0.60 - hold_ratio) * 300.0
    if float(metrics.get("hold_oscillation_count", 0.0)) >= 2.0:
        score += 120.0
    if float(metrics.get("hold_drift", 0.0)) > max(abs(float(metrics.get("steady_state_error", 0.0))), 1.0):
        score += 80.0
    if cancelled_candidate:
        score += 250.0
    if aborted:
        score += 500.0
    if float(metrics.get("invalid_ratio", 0.0)) >= 0.5:
        score += 200.0
    if float(metrics.get("oscillation_rate", 0.0)) >= 0.5:
        score += 150.0
    return float(score)


def propose_coordinate_candidate(
    base_pid: tuple[float, float, float],
    axis_index: int,
    axis_direction: float,
    *,
    step_kp: float,
    step_ki: float,
    step_kd: float,
    kp_max: float,
    ki_max: float,
    kd_max: float,
) -> tuple[tuple[float, float, float], int, float, float]:
    """Adjust exactly one PID term from the current base point."""
    values = [float(base_pid[0]), float(base_pid[1]), float(base_pid[2])]
    step_sizes = [float(step_kp), float(step_ki), float(step_kd)]
    max_values = [float(kp_max), float(ki_max), float(kd_max)]

    used_axis = int(axis_index) % 3
    direction = 1.0 if axis_direction >= 0 else -1.0
    delta = direction * step_sizes[used_axis]
    proposed_value = float(np.clip(values[used_axis] + delta, 0.0, max_values[used_axis]))

    if np.isclose(proposed_value, values[used_axis]):
        direction *= -1.0
        delta = direction * step_sizes[used_axis]
        proposed_value = float(np.clip(values[used_axis] + delta, 0.0, max_values[used_axis]))

    values[used_axis] = proposed_value
    actual_delta = values[used_axis] - base_pid[used_axis]
    return (values[0], values[1], values[2]), used_axis, direction, float(actual_delta)


def candidate_is_safe(
    metrics: dict,
    *,
    cancelled_candidate: bool,
    aborted: bool,
    max_invalid_ratio: float,
    max_oscillation_rate: float,
) -> bool:
    """Gate Bayesian search until the warmup has produced stable candidates."""
    if cancelled_candidate or aborted:
        return False
    if float(metrics.get("invalid_ratio", 1.0)) > max_invalid_ratio:
        return False
    if float(metrics.get("oscillation_rate", 1.0)) > max_oscillation_rate:
        return False
    return True


def candidate_is_good(
    metrics: dict,
    score: float,
    *,
    cancelled_candidate: bool,
    aborted: bool,
    baseline_score: float | None,
    max_invalid_ratio: float,
    max_oscillation_rate: float,
    max_score_factor: float,
) -> bool:
    """Require stability plus a score near or better than the baseline."""
    if not candidate_is_safe(
        metrics,
        cancelled_candidate=cancelled_candidate,
        aborted=aborted,
        max_invalid_ratio=max_invalid_ratio,
        max_oscillation_rate=max_oscillation_rate,
    ):
        return False
    if baseline_score is None or baseline_score <= 0:
        return True
    return float(score) <= (float(baseline_score) * float(max_score_factor))


def compute_bootstrap_axis_status(
    safe_points: list[tuple[float, float, float]],
    *,
    min_points_per_axis: int,
    min_span_kp: float,
    min_span_ki: float,
    min_span_kd: float,
) -> list[dict]:
    """Summarise bootstrap coverage progress separately for each PID axis."""
    required_spans = (float(min_span_kp), float(min_span_ki), float(min_span_kd))
    if not safe_points:
        return [
            {
                "axis_index": idx,
                "axis_name": AXIS_NAMES[idx],
                "distinct_safe_values": 0,
                "safe_span": 0.0,
                "required_distinct_values": int(min_points_per_axis),
                "required_safe_span": float(required_spans[idx]),
                "distinct_deficit": int(min_points_per_axis),
                "span_deficit": float(required_spans[idx]),
                "distinct_coverage": 0.0,
                "span_coverage": 0.0,
                "deficit_score": float(int(min_points_per_axis) + 1.0),
                "coverage_ratio": 0.0,
                "bootstrap_complete": False,
                "complete": False,
            }
            for idx in range(3)
        ]

    safe_arr = np.asarray(safe_points, dtype=float)
    statuses: list[dict] = []
    for idx, required_span in enumerate(required_spans):
        axis_values = np.round(safe_arr[:, idx], 6)
        distinct_safe_values = int(len(np.unique(axis_values)))
        safe_span = float(np.max(safe_arr[:, idx]) - np.min(safe_arr[:, idx])) if safe_arr.size else 0.0
        distinct_deficit = max(0, int(min_points_per_axis) - distinct_safe_values)
        span_deficit = max(0.0, float(required_span) - safe_span)
        distinct_ratio = (
            min(1.0, distinct_safe_values / float(max(1, int(min_points_per_axis))))
            if int(min_points_per_axis) > 0
            else 1.0
        )
        span_ratio = min(1.0, safe_span / float(max(required_span, 1e-9))) if required_span > 0 else 1.0
        coverage_ratio = min(distinct_ratio, span_ratio)
        bootstrap_complete = bool(distinct_deficit == 0 and span_deficit <= 1e-9)
        deficit_score = float(distinct_deficit) + (float(span_deficit) / float(max(required_span, 1e-9)))
        statuses.append(
            {
                "axis_index": idx,
                "axis_name": AXIS_NAMES[idx],
                "distinct_safe_values": distinct_safe_values,
                "safe_span": safe_span,
                "required_distinct_values": int(min_points_per_axis),
                "required_safe_span": float(required_span),
                "distinct_deficit": distinct_deficit,
                "span_deficit": span_deficit,
                "distinct_coverage": float(distinct_ratio),
                "span_coverage": float(span_ratio),
                "deficit_score": float(deficit_score),
                "coverage_ratio": float(coverage_ratio),
                "bootstrap_complete": bootstrap_complete,
                "complete": bootstrap_complete,
            }
        )
    return statuses


def bootstrap_axes_complete(axis_statuses: list[dict]) -> bool:
    """Return True when every PID axis has met bootstrap safe-coverage requirements."""
    return bool(axis_statuses) and all(bool(status.get("bootstrap_complete", False)) for status in axis_statuses)


def choose_bootstrap_axis(
    base_pid: tuple[float, float, float],
    *,
    safe_points: list[tuple[float, float, float]],
    axis_directions: list[float],
    preferred_axis_index: int,
    step_kp: float,
    step_ki: float,
    step_kd: float,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    min_points_per_axis: int,
    min_span_kp: float,
    min_span_ki: float,
    min_span_kd: float,
) -> tuple[int, list[dict]]:
    """Choose the next bootstrap axis, preferring incomplete coverage first."""
    axis_statuses = compute_bootstrap_axis_status(
        safe_points,
        min_points_per_axis=min_points_per_axis,
        min_span_kp=min_span_kp,
        min_span_ki=min_span_ki,
        min_span_kd=min_span_kd,
    )

    probeable_axes: list[tuple[dict, bool]] = []
    for status in axis_statuses:
        candidate, _, _, actual_delta = propose_coordinate_candidate(
            base_pid,
            axis_index=status["axis_index"],
            axis_direction=axis_directions[status["axis_index"] % 3],
            step_kp=step_kp,
            step_ki=step_ki,
            step_kd=step_kd,
            kp_max=kp_max,
            ki_max=ki_max,
            kd_max=kd_max,
        )
        _ = candidate
        if abs(float(actual_delta)) > 1e-9:
            probeable_axes.append(
                (status, int(status["axis_index"]) == (int(preferred_axis_index) % 3))
            )

    if not probeable_axes:
        return int(preferred_axis_index) % 3, axis_statuses

    incomplete_axes = [
        (status, is_preferred_axis)
        for status, is_preferred_axis in probeable_axes
        if not bool(status.get("bootstrap_complete", status.get("complete", False)))
    ]
    ranked_axes = incomplete_axes if incomplete_axes else probeable_axes

    def rank_key(item: tuple[dict, bool]) -> tuple[float, float, float, int, int]:
        status, is_preferred_axis = item
        return (
            -float(status.get("deficit_score", 0.0)),
            float(status.get("coverage_ratio", 1.0)),
            float(status.get("span_coverage", 1.0)),
            0 if is_preferred_axis else 1,
            int(status["axis_index"]),
        )

    ranked_axes.sort(key=rank_key)
    return int(ranked_axes[0][0]["axis_index"]), axis_statuses


def assess_bayes_region(
    safe_points: list[tuple[float, float, float]],
    good_points: list[tuple[float, float, float]],
    *,
    min_safe_candidates: int,
    min_points_per_axis: int,
    min_good_candidates: int,
    min_span_kp: float,
    min_span_ki: float,
    min_span_kd: float,
) -> dict:
    """Decide whether the warmup has mapped a usable local region for BO."""
    axis_statuses = compute_bootstrap_axis_status(
        safe_points,
        min_points_per_axis=min_points_per_axis,
        min_span_kp=min_span_kp,
        min_span_ki=min_span_ki,
        min_span_kd=min_span_kd,
    )
    if not safe_points:
        return {
            "ready": False,
            "reason": "no safe candidates yet",
            "unique_counts": (0, 0, 0),
            "spans": (0.0, 0.0, 0.0),
            "axis_statuses": axis_statuses,
        }

    unique_counts = tuple(int(status["distinct_safe_values"]) for status in axis_statuses)
    spans = tuple(float(status["safe_span"]) for status in axis_statuses)
    required_spans = (float(min_span_kp), float(min_span_ki), float(min_span_kd))

    if len(safe_points) < int(min_safe_candidates):
        return {
            "ready": False,
            "reason": f"only {len(safe_points)} safe candidates",
            "unique_counts": unique_counts,
            "spans": spans,
            "axis_statuses": axis_statuses,
        }

    if len(good_points) < int(min_good_candidates):
        return {
            "ready": False,
            "reason": f"only {len(good_points)} good candidates",
            "unique_counts": unique_counts,
            "spans": spans,
            "axis_statuses": axis_statuses,
        }

    if not bootstrap_axes_complete(axis_statuses):
        incomplete_status = next(
            (
                status
                for status in axis_statuses
                if not bool(status.get("bootstrap_complete", status.get("complete", False)))
            ),
            None,
        )
        if incomplete_status is not None:
            return {
                "ready": False,
                "reason": (
                    f"{incomplete_status['axis_name']} coverage "
                    f"{incomplete_status['distinct_safe_values']}/{incomplete_status['required_distinct_values']} "
                    f"| span {incomplete_status['safe_span']:.4f}/{incomplete_status['required_safe_span']:.4f}"
                ),
                "unique_counts": unique_counts,
                "spans": spans,
                "axis_statuses": axis_statuses,
            }

    return {
        "ready": True,
        "reason": "safe local region established",
        "unique_counts": unique_counts,
        "spans": spans,
        "axis_statuses": axis_statuses,
    }


def build_bayes_search_space(
    safe_pids: list[tuple[float, float, float]],
    *,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    pad_kp: float,
    pad_ki: float,
    pad_kd: float,
) -> list[BayesSpaceDim]:
    """Build a local Bayesian search box around the safe region from coordinate search."""
    if not safe_pids:
        return [
            Real(0.0, kp_max, name="kp"),
            Real(0.0, ki_max, name="ki"),
            Real(0.0, kd_max, name="kd"),
        ]

    mins = np.min(np.asarray(safe_pids, dtype=float), axis=0)
    maxs = np.max(np.asarray(safe_pids, dtype=float), axis=0)
    pads = np.asarray([pad_kp, pad_ki, pad_kd], dtype=float)
    bounds_max = np.asarray([kp_max, ki_max, kd_max], dtype=float)
    lower = np.clip(mins - pads, 0.0, bounds_max)
    upper = np.clip(maxs + pads, 0.0, bounds_max)

    for idx in range(3):
        if upper[idx] <= lower[idx]:
            upper[idx] = min(bounds_max[idx], lower[idx] + max(pads[idx], 1e-3))
            lower[idx] = max(0.0, upper[idx] - max(pads[idx], 1e-3))

    return [
        Real(float(lower[0]), float(upper[0]), name="kp"),
        Real(float(lower[1]), float(upper[1]), name="ki"),
        Real(float(lower[2]), float(upper[2]), name="kd"),
    ]


def filter_seed_points_for_space(
    points: list[tuple[float, float, float]],
    scores: list[float],
    space: list[BayesSpaceDim],
) -> tuple[list[list[float]], list[float]]:
    """Keep only warmup observations that fit inside the Bayesian local box."""
    filtered_points: list[list[float]] = []
    filtered_scores: list[float] = []
    for point, score in zip(points, scores):
        if all(dim.low <= value <= dim.high for dim, value in zip(space, point)):
            filtered_points.append([float(point[0]), float(point[1]), float(point[2])])
            filtered_scores.append(float(score))
    return filtered_points, filtered_scores


def format_readiness_status(
    *,
    region_status: dict,
    safe_count: int,
    safe_target: int,
    good_count: int,
    good_target: int,
    unique_counts: tuple[int, int, int],
    per_axis_target: int,
    spans: tuple[float, float, float],
    span_targets: tuple[float, float, float],
    warmup_trials_done: int | None = None,
    warmup_trials_target: int | None = None,
) -> str:
    """Build a short GUI checklist for post-warmup optimisation readiness."""
    def mark(done: bool) -> str:
        return "[x]" if done else "[ ]"

    tail = f"Blocked by: {region_status['reason']}"
    if bool(region_status.get("ready")):
        tail = "Region ready. Surrogate-guided optimisation can start now."

    return (
        "BO readiness:\n"
        f"{mark(safe_count >= safe_target)} Safe candidates: {safe_count}/{safe_target}\n"
        f"{mark(good_count >= good_target)} Good candidates: {good_count}/{good_target}\n"
        f"{mark(unique_counts[0] >= per_axis_target)} Kp coverage: {unique_counts[0]}/{per_axis_target} | "
        f"span {spans[0]:.3f}/{span_targets[0]:.3f}\n"
        f"{mark(unique_counts[1] >= per_axis_target)} Ki coverage: {unique_counts[1]}/{per_axis_target} | "
        f"span {spans[1]:.3f}/{span_targets[1]:.3f}\n"
        f"{mark(unique_counts[2] >= per_axis_target)} Kd coverage: {unique_counts[2]}/{per_axis_target} | "
        f"span {spans[2]:.3f}/{span_targets[2]:.3f}\n"
        f"{tail}"
    )


def format_warmup_change_message(
    base_pid: tuple[float, float, float] | None,
    candidate_pid: tuple[float, float, float] | None,
    used_axis: int | None,
    candidate_delta: float,
) -> str:
    """Describe the current warmup move for the monitor."""
    if candidate_pid is None:
        return "Warmup change: waiting for first candidate"
    if base_pid is None or used_axis is None:
        return "Warmup change: baseline trial using current hardware PID (no warmup delta)"
    return (
        "Warmup change: "
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
    """Summarise whether the last warmup candidate passed and why it failed if not."""
    reasons: list[str] = []

    if cancelled_candidate:
        reasons.append(cancel_reason or "remaining repeats cancelled")
    if aborted:
        reasons.append("trial aborted by safety condition")

    per_repeat_reasons = []
    for meta in per_test_meta:
        reason = str(meta.get("reason", "")).strip()
        if reason:
            per_repeat_reasons.append(reason)
    primary_repeat_reason = list(dict.fromkeys(per_repeat_reasons))[0] if per_repeat_reasons else ""

    invalid_ratio = float(metrics.get("invalid_ratio", 0.0))
    if invalid_ratio > float(safe_invalid_ratio):
        detail = f"invalid ratio {invalid_ratio:.2f} > {float(safe_invalid_ratio):.2f}"
        if primary_repeat_reason:
            detail = f"{detail} ({primary_repeat_reason})"
        reasons.append(detail)

    oscillation_rate = float(metrics.get("oscillation_rate", 0.0))
    if oscillation_rate > float(safe_oscillation_rate):
        reasons.append(f"oscillation {oscillation_rate:.2f} > {float(safe_oscillation_rate):.2f}")

    if baseline_score is not None and baseline_score > 0:
        good_limit = float(baseline_score) * float(good_score_factor)
        if score > good_limit:
            reasons.append(f"score {score:.2f} > good threshold {good_limit:.2f}")

    if reasons:
        return "Previous warmup result: failed - " + "; ".join(reasons)

    return (
        "Previous warmup result: passed - "
        f"score={score:.2f}, invalid={invalid_ratio:.2f}, osc={oscillation_rate:.2f}"
    )


def format_phase_display(mode: str) -> str:
    """Format a concise, prominent phase label for the runtime monitor."""
    if mode == "validation":
        return "VALIDATION"
    if mode.startswith("surrogate"):
        return "OPTIMISATION (SURROGATE)"
    if mode == "bo":
        return "OPTIMISATION (BO)"
    if mode == "fallback":
        return "OPTIMISATION (FALLBACK)"
    return "BOOTSTRAP"


def format_candidate_source(mode: str) -> str:
    """Format the current candidate source label for the runtime monitor."""
    if mode.startswith("surrogate"):
        return "surrogate"
    if mode == "bo":
        return "BO"
    if mode == "fallback":
        return "fallback"
    return "bootstrap"


def counted_trial_totals(*, mode: str, n_trials: int, validation_trials: int) -> tuple[int | None, int, int, int]:
    """Return upcoming phase labels and split counters for the monitor."""
    if mode == "validation":
        return int(validation_trials), 0, 0, 1
    if mode == "warmup":
        return None, 1, 0, 0
    return int(n_trials), 0, 1, 0


def main():
    """Parse inputs, run selected action, and manage full tuning workflow."""
    import argparse

    # Runtime arguments for serial connection, tuning bounds, and target output.
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
        help="Hard cap on bootstrap safe-range discovery trials; these do not count toward --iters",
    )
    ap.add_argument(
        "--bayes-min-safe-trials",
        type=int,
        default=4,
        help="Minimum number of safe warmup trials required before Bayesian optimisation starts",
    )
    ap.add_argument(
        "--bayes-region-min-points-per-axis",
        type=int,
        default=3,
        help="Minimum number of distinct safe values required on each PID axis before Bayesian optimisation starts",
    )
    ap.add_argument(
        "--bayes-region-min-good-candidates",
        type=int,
        default=2,
        help="Minimum number of stable, reasonably accurate warmup candidates required before Bayesian optimisation starts",
    )
    ap.add_argument(
        "--bayes-region-good-score-factor",
        type=float,
        default=1.05,
        help="Warmup candidate counts as good if its score is at most this multiple of the baseline score",
    )
    ap.add_argument(
        "--bayes-region-min-span-kp",
        type=float,
        default=0.05,
        help="Minimum safe Kp span required before Bayesian optimisation starts",
    )
    ap.add_argument(
        "--bayes-region-min-span-ki",
        type=float,
        default=0.05,
        help="Minimum safe Ki span required before Bayesian optimisation starts",
    )
    ap.add_argument(
        "--bayes-region-min-span-kd",
        type=float,
        default=0.01,
        help="Minimum safe Kd span required before Bayesian optimisation starts",
    )
    ap.add_argument(
        "--bayes-safe-invalid-ratio",
        type=float,
        default=0.20,
        help="Maximum invalid ratio a warmup trial can have and still count as safe for Bayesian startup",
    )
    ap.add_argument(
        "--bayes-safe-oscillation-rate",
        type=float,
        default=0.30,
        help="Maximum oscillation rate a warmup trial can have and still count as safe for Bayesian startup",
    )
    ap.add_argument(
        "--repeat-cancel-osc-threshold",
        type=float,
        default=0.80,
        help="Cancel remaining repeats for a PID candidate when oscillation rate meets or exceeds this value",
    )
    ap.add_argument(
        "--repeat-cancel-score-regression-pct",
        type=float,
        default=8.0,
        help="Cancel remaining repeats only when the repeat score regresses by at least this percentage",
    )
    ap.add_argument(
        "--lock-growth-after-improve-pct",
        type=float,
        default=20.0,
        help="If best improvement >= this, stop increasing step sizes on misses",
    )
    ap.add_argument("--early-stop-patience", type=int, default=12, help="Stop after N non-improving trials")
    ap.add_argument(
        "--retest-best-every",
        type=int,
        default=0,
        help="Every N trials, re-run current best PID for verification (0 disables)",
    )
    ap.add_argument(
        "--refine-activate-improve-pct",
        type=float,
        default=25.0,
        help="Enable local refinement bounds after this best-improvement percentage",
    )
    ap.add_argument("--refine-radius-kp", type=float, default=0.2, help="Refinement radius around best Kp")
    ap.add_argument("--refine-radius-ki", type=float, default=0.2, help="Refinement radius around best Ki")
    ap.add_argument("--refine-radius-kd", type=float, default=0.05, help="Refinement radius around best Kd")
    ap.add_argument("--no-gui", action="store_true", help="Disable launch GUI and use console prompts")
    ap.add_argument("--power-csv", default="tuning_power_readings.csv", help="CSV file for graphing power readings")
    ap.add_argument("--test-duration-s", type=float, default=12.0, help="Seconds per individual laser test")
    ap.add_argument(
        "--warmup-repeats",
        type=int,
        default=3,
        help="Number of repeated tests per candidate during candidate gathering",
    )
    ap.add_argument(
        "--bo-repeats",
        type=int,
        default=5,
        help="Number of repeated tests per candidate during Bayesian optimisation",
    )
    ap.add_argument("--frequency-khz", type=int, default=0, help="Laser frequency in kHz for the startup program command")
    ap.add_argument(
        "--surrogate-model",
        choices=("extra_trees", "random_forest", "none"),
        default="extra_trees",
        help="Lightweight surrogate model used after warmup",
    )
    ap.add_argument(
        "--surrogate-min-samples",
        type=int,
        default=8,
        help="Minimum completed candidates before surrogate-guided search is enabled",
    )
    ap.add_argument(
        "--surrogate-retrain-every",
        type=int,
        default=1,
        help="Refit the surrogate every N completed candidates",
    )
    ap.add_argument(
        "--surrogate-pool-size",
        type=int,
        default=36,
        help="Candidate pool size scored by the surrogate before each proposal",
    )
    ap.add_argument(
        "--surrogate-explore-prob",
        type=float,
        default=0.25,
        help="Exploration probability during surrogate-guided search",
    )
    ap.add_argument(
        "--surrogate-jitter-scale",
        type=float,
        default=1.25,
        help="Scale factor for surrogate proposal jitter around safe/best points",
    )
    ap.add_argument(
        "--bo-refine-trials",
        type=int,
        default=4,
        help="Optional BO-style refinement trials after surrogate search (0 disables)",
    )
    ap.add_argument(
        "--validation-trials",
        type=int,
        default=1,
        help="Optional validation re-tests of the final best candidate after optimisation (does not count toward --iters)",
    )
    ap.add_argument(
        "--validation-repeats",
        type=int,
        default=5,
        help="Number of repeated tests per validation candidate",
    )
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
        except Exception as e:
            log(f"Warning: Could not open serial port for startup defaults: {e}")
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
            except RuntimeError as e:
                log(f"Graph tool error: {e}")
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
            log(
                f"Configured counted optimisation trials after bootstrap: {n_trials} "
                "(bootstrap and validation are extra and logged separately)"
            )
            if test_duration_s is None:
                test_duration_s = float(args.test_duration_s)
            log(f"Configured per-test duration: {test_duration_s:.2f}s")
            log(
                f"Configured repeats: warmup={int(args.warmup_repeats)}, "
                f"BO={int(args.bo_repeats)}"
            )
            if frequency_khz is None:
                frequency_khz = int(default_frequency_khz)
            log(f"Configured frequency: {frequency_khz} kHz")
            if monitor is not None:
                monitor.set_target(desired_output)
                monitor.set_phase("Phase: Bootstrap / safe-range discovery")
                monitor.set_status("Sending startup program command")

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

            # Keep a trial-by-trial record for later review.
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
            observed_points: list[tuple[float, float, float]] = []
            observed_scores: list[float] = []
            surrogate_training_rows: list[dict] = []
            surrogate = OnlineSurrogateModel(args.surrogate_model, random_state=42)
            surrogate_active = False
            rng = random.Random(42)
            region_status = {
                "ready": False,
                "reason": "no safe candidates yet",
                "unique_counts": (0, 0, 0),
                "spans": (0.0, 0.0, 0.0),
                "axis_statuses": [],
            }
            if monitor is not None:
                bootstrap_target = max(1, int(args.coordinate_warmup_trials))
                monitor.set_phase("BOOTSTRAP")
                monitor.set_candidate_source("bootstrap")
                monitor.set_trial_counters(bootstrap_used=0, optimisation_used=0, validation_used=0)
                monitor.set_axis_coverage(region_status["axis_statuses"])
                monitor.set_best_candidate(kp=None, ki=None, kd=None, score=None)
                monitor.set_warmup_counter(f"Bootstrap counter: 0/{bootstrap_target} completed")
                monitor.set_warmup_change("Bootstrap change: baseline trial using current hardware PID (no bootstrap delta)")
                monitor.set_previous_warmup_result("Previous bootstrap result: none yet")
                monitor.set_readiness(
                    format_readiness_status(
                        region_status=region_status,
                        safe_count=0,
                        safe_target=int(args.bayes_min_safe_trials),
                        good_count=0,
                        good_target=int(args.bayes_region_min_good_candidates),
                        unique_counts=(0, 0, 0),
                        per_axis_target=int(args.bayes_region_min_points_per_axis),
                        spans=(0.0, 0.0, 0.0),
                        span_targets=(
                            float(args.bayes_region_min_span_kp),
                            float(args.bayes_region_min_span_ki),
                            float(args.bayes_region_min_span_kd),
                        ),
                        warmup_trials_done=0,
                        warmup_trials_target=bootstrap_target,
                    )
                )

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
                            **repeat_feature,
                        }
                    )

                for test_idx, (test_powers, test_times, test_meta) in enumerate(
                    zip(per_test_powers, per_test_times, per_test_meta),
                    start=1,
                ):
                    if test_powers.size == 0:
                        continue
                    if test_times.size == test_powers.size:
                        time_vals = test_times.tolist()
                    else:
                        time_vals = list(range(int(test_powers.size)))
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
                predicted_score: float | None = None,
                surrogate_enabled: bool = False,
            ) -> float:
                nonlocal trial_index, bootstrap_trial_count, optimisation_trial_count, validation_trial_count
                nonlocal surrogate_trial_count, bo_trial_count, fallback_trial_count
                nonlocal baseline_score, best_score_seen, best_pid, best_metrics, last_applied
                nonlocal no_improve_count, step_kp, step_ki, step_kd, axis_index, surrogate_active
                phase = mode_to_phase(mode)
                is_warmup_mode = mode == "warmup"
                is_surrogate_mode = mode.startswith("surrogate")
                is_bo_mode = mode == "bo"
                is_validation_mode = mode == "validation"
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
                if is_warmup_mode:
                    phase_repeats = max(1, int(args.warmup_repeats))
                elif is_validation_mode:
                    phase_repeats = max(1, int(args.validation_repeats))
                else:
                    phase_repeats = max(1, int(args.bo_repeats))
                log(
                    f"{display_phase_name} trial {display_phase_index}"
                    + (f"/{display_phase_total}" if display_phase_total is not None else "")
                    + f" (overall {trial_index + 1})"
                )
                if monitor is not None:
                    monitor.set_phase(format_phase_display(mode))
                    if not is_validation_mode:
                        monitor.set_candidate_source(format_candidate_source(mode))
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
                        monitor.set_warmup_change("Bootstrap change: safe-range discovery in progress")
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
                    repeat_cancel_osc_threshold=args.repeat_cancel_osc_threshold,
                    repeat_cancel_score_regression_pct=args.repeat_cancel_score_regression_pct,
                )

                used_kp, used_ki, used_kd = kp, ki, kd
                if trial_index == 0:
                    if current_pid is not None:
                        used_kp = float(current_pid["pw_kp"])
                        used_ki = float(current_pid["pw_ki"])
                        used_kd = float(current_pid["pw_kd"])
                        log(
                            "Stored initial laser PID values for baseline trial: "
                            f"kp={used_kp:.4f}, ki={used_ki:.4f}, kd={used_kd:.4f}"
                        )
                        last_applied = (used_kp, used_ki, used_kd)
                    else:
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
                    if improved:
                        best_pid = (used_kp, used_ki, used_kd)
                        best_metrics = dict(metrics)
                        no_improve_count = 0
                        step_kp = max(min_step_kp, step_kp * float(args.step_shrink_factor))
                        step_ki = max(min_step_ki, step_ki * float(args.step_shrink_factor))
                        step_kd = max(min_step_kd, step_kd * float(args.step_shrink_factor))
                    else:
                        no_improve_count += 1
                        if used_axis is not None:
                            axis_directions[used_axis] *= -1.0
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
                    region_status = assess_bayes_region(
                        safe_trial_points,
                        good_trial_points,
                        min_safe_candidates=args.bayes_min_safe_trials,
                        min_points_per_axis=args.bayes_region_min_points_per_axis,
                        min_good_candidates=args.bayes_region_min_good_candidates,
                        min_span_kp=args.bayes_region_min_span_kp,
                        min_span_ki=args.bayes_region_min_span_ki,
                        min_span_kd=args.bayes_region_min_span_kd,
                    )
                    log(
                        "Candidate region status -> "
                        f"ready={region_status['ready']}, "
                        f"reason={region_status['reason']}, "
                        f"safe={len(safe_trial_points)}, "
                        f"good={len(good_trial_points)}, "
                        f"unique={region_status['unique_counts']}, "
                        f"spans=({region_status['spans'][0]:.4f},"
                        f"{region_status['spans'][1]:.4f},"
                        f"{region_status['spans'][2]:.4f})"
                    )
                    if monitor is not None:
                        bootstrap_done = bootstrap_trial_count + 1
                        bootstrap_target = max(1, int(args.coordinate_warmup_trials))
                        remaining = max(0, bootstrap_target - bootstrap_done)
                        monitor.set_axis_coverage(region_status["axis_statuses"])
                        monitor.set_warmup_counter(
                            f"Bootstrap counter: {bootstrap_done}/{bootstrap_target} completed | {remaining} remaining"
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
                                region_status=region_status,
                                safe_count=len(safe_trial_points),
                                safe_target=int(args.bayes_min_safe_trials),
                                good_count=len(good_trial_points),
                                good_target=int(args.bayes_region_min_good_candidates),
                                unique_counts=tuple(region_status["unique_counts"]),
                                per_axis_target=int(args.bayes_region_min_points_per_axis),
                                spans=tuple(region_status["spans"]),
                                span_targets=(
                                    float(args.bayes_region_min_span_kp),
                                    float(args.bayes_region_min_span_ki),
                                    float(args.bayes_region_min_span_kd),
                                ),
                                warmup_trials_done=bootstrap_done,
                                warmup_trials_target=bootstrap_target,
                            )
                        )
                        if bool(region_status.get("ready")):
                            monitor.set_phase("BOOTSTRAP")
                            monitor.set_progress("Surrogate guidance ready | preparing model")
                            monitor.set_warmup_change("Bootstrap change: complete")

                    if args.surrogate_model != "none" and (len(surrogate_training_rows) % max(1, int(args.surrogate_retrain_every)) == 0):
                        surrogate_active = surrogate.fit(
                            surrogate_training_rows,
                            min_samples=max(
                                int(args.surrogate_min_samples),
                                int(args.bayes_min_safe_trials),
                            ),
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
                    f"Result -> score={score:.2f}, "
                    f"improve={improve_vs_base_pct:.2f}%, "
                    f"best_improve={best_improve_vs_base_pct:.2f}%, "
                    f"no_improve={no_improve_count}, "
                    f"step=({step_kp:.4f},{step_ki:.4f},{step_kd:.4f}), "
                    f"start_err={metrics['start_error']:.5f}, "
                    f"track_err={metrics['track_error']:.5f}, "
                    f"dev={metrics['deviation']:.5f}, "
                    f"max_err={metrics['max_error']:.5f}, "
                    f"strict_bad={metrics['strict_bad_rate']:.5f}, "
                    f"osc={metrics['oscillation_rate']:.5f}, "
                    f"overshoot={metrics['overshoot_pct']:.3f}, "
                    f"settle={metrics['settling_time_s']:.3f}, "
                    f"sse={metrics['steady_state_error']:.5f}, "
                    f"hold={metrics.get('hold_quality', math.nan):.3f}, "
                    f"hold_tol={metrics.get('hold_time_in_tolerance_ratio', math.nan):.3f}, "
                    f"hold_var={metrics.get('hold_variance', math.nan):.5f}, "
                    f"iae={metrics['iae']:.5f}, "
                    f"invalid={metrics['invalid_ratio']:.3f}, "
                    f"repeat={metrics['repeatability']:.5f}, "
                    f"cancelled={cancelled_candidate}, "
                    f"aborted={aborted}"
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
                    axis_state = next(
                        (status for status in axis_statuses if int(status["axis_index"]) == int(used_axis)),
                        None,
                    )
                    coverage_summary = ""
                    if axis_state is not None:
                        coverage_summary = (
                            f"coverage={axis_state['distinct_safe_values']}/{axis_state['required_distinct_values']}, "
                            f"span={axis_state['safe_span']:.4f}/{axis_state['required_safe_span']:.4f}, "
                            f"complete={axis_state['complete']}, "
                        )
                    log(
                        "Coordinate candidate -> "
                        f"base=({base_pid[0]:.4f},{base_pid[1]:.4f},{base_pid[2]:.4f}), "
                        f"axis={AXIS_NAMES[used_axis]}, delta={candidate_delta:+.4f}, "
                        f"{coverage_summary}"
                        f"candidate=({kp:.4f},{ki:.4f},{kd:.4f})"
                    )
                    if monitor is not None:
                        monitor.set_warmup_change(
                            format_warmup_change_message(
                                base_pid,
                                (kp, ki, kd),
                                used_axis,
                                candidate_delta,
                            )
                        )
                else:
                    kp, ki, kd = 0.0, 0.0, 0.0
                    if monitor is not None:
                        monitor.set_warmup_change(
                            format_warmup_change_message(None, (kp, ki, kd), None, 0.0)
                        )

                score = evaluate_candidate(kp, ki, kd, mode="warmup", used_axis=used_axis)
                if used_axis is not None:
                    axis_index = (used_axis + 1) % 3
                return score

            def run_surrogate_trial() -> float:
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
                    step_kp=step_kp,
                    step_ki=step_ki,
                    step_kd=step_kd,
                    kp_max=args.kp_max,
                    ki_max=args.ki_max,
                    kd_max=args.kd_max,
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
                base_pid = best_pid if best_pid is not None else last_applied
                if base_pid is None:
                    base_pid = (0.0, 0.0, 0.0)
                (kp, ki, kd), used_axis, _, candidate_delta = propose_coordinate_candidate(
                    base_pid,
                    axis_index=axis_index,
                    axis_direction=axis_directions[axis_index % 3],
                    step_kp=step_kp,
                    step_ki=step_ki,
                    step_kd=step_kd,
                    kp_max=args.kp_max,
                    ki_max=args.ki_max,
                    kd_max=args.kd_max,
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
                score = evaluate_candidate(kp, ki, kd, mode="fallback", used_axis=used_axis)
                axis_index = (used_axis + 1) % 3
                return score

            log("Starting bootstrap safe-range discovery")
            if monitor is not None:
                monitor.set_phase(format_phase_display("warmup"))
                monitor.set_candidate_source(format_candidate_source("warmup"))
            try:
                warmup_target = max(1, int(args.coordinate_warmup_trials))
                max_bootstrap_trials = max(warmup_target, int(args.max_bootstrap_trials))
                while bootstrap_trial_count < max_bootstrap_trials and not bool(region_status["ready"]):
                    run_coordinate_trial()

                if bool(region_status["ready"]):
                    no_improve_count = 0
                    surrogate_budget = max(0, int(n_trials) - max(0, int(args.bo_refine_trials)))
                    bo_budget = max(0, min(int(args.bo_refine_trials), int(n_trials)))
                    log(
                        f"Bootstrap complete after {bootstrap_trial_count} trials; "
                        f"planned phase budgets -> surrogate={surrogate_budget}, bo_refine={bo_budget}"
                    )
                    if monitor is not None:
                        monitor.set_phase(format_phase_display("warmup"))
                        monitor.set_warmup_counter("")
                        monitor.set_readiness("Optimisation readiness:\n[x] Bootstrap complete. Switching to counted optimisation.")

                    if args.surrogate_model != "none":
                        surrogate_active = surrogate.fit(
                            surrogate_training_rows,
                            min_samples=max(
                                int(args.surrogate_min_samples),
                                int(args.bayes_min_safe_trials),
                            ),
                        )
                        if surrogate_active:
                            log(
                                f"Surrogate ready -> model={args.surrogate_model}, "
                                f"samples={surrogate.last_fit_count}"
                            )
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
                        )
                        seed_points, seed_scores = filter_seed_points_for_space(
                            observed_points,
                            observed_scores,
                            bayes_space,
                        )
                        gp_minimize(
                            lambda x: evaluate_candidate(float(x[0]), float(x[1]), float(x[2]), mode="bo"),
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
                        log("Using safe fallback coordinate search for remaining optimisation budget")
                        run_fallback_trial()
                        remaining_after_bo -= 1
                else:
                    log(
                        "Optimisation phases skipped: "
                        f"no viable safe region found after {bootstrap_trial_count} bootstrap trials "
                        f"({region_status['reason']})"
                    )
                    if monitor is not None:
                        monitor.set_phase(format_phase_display("warmup"))
                        monitor.set_candidate_source(format_candidate_source("warmup"))
                        monitor.set_progress(
                            f"Bootstrap stopped after {bootstrap_trial_count} trials | {region_status['reason']}"
                        )
                        monitor.set_warmup_counter("")
                        monitor.set_readiness(
                            format_readiness_status(
                                region_status=region_status,
                                safe_count=len(safe_trial_points),
                                safe_target=int(args.bayes_min_safe_trials),
                                good_count=len(good_trial_points),
                                good_target=int(args.bayes_region_min_good_candidates),
                                unique_counts=tuple(region_status["unique_counts"]),
                                per_axis_target=int(args.bayes_region_min_points_per_axis),
                                spans=tuple(region_status["spans"]),
                                span_targets=(
                                    float(args.bayes_region_min_span_kp),
                                    float(args.bayes_region_min_span_ki),
                                    float(args.bayes_region_min_span_kd),
                                ),
                                warmup_trials_done=bootstrap_trial_count,
                                warmup_trials_target=warmup_target,
                            )
                        )
            except EarlyStopOptimization as e:
                log(f"Early stop: {e}")

            if (
                bool(region_status["ready"])
                and best_pid is not None
                and int(args.validation_trials) > 0
                and optimisation_trial_count > 0
            ):
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
            if best_score_seen < float("inf"):
                log(f"Best score={best_score_seen:.3f}")
            log(
                "Phase counts -> "
                f"bootstrap={bootstrap_trial_count}, "
                f"optimisation={optimisation_trial_count}, "
                f"validation={validation_trial_count}"
            )

            with open("tuning_history.csv", "w", newline="") as f:
                fieldnames = list(history[0].keys()) if history else [
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
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(history)
            log("Saved tuning_history.csv")

            with open("tuning_trace_features.csv", "w", newline="") as f:
                fieldnames = list(trace_feature_rows[0].keys()) if trace_feature_rows else [
                    "trial_index",
                    "phase",
                    "repeat_index",
                    "phase_mode",
                    "kp",
                    "ki",
                    "kd",
                ]
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

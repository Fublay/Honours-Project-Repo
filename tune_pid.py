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

import numpy as np
import serial
from skopt import gp_minimize
from skopt.space import Real

import laser_command_ids as CMD
from pipeline.data_collector import collect_trial_data
from protocol.reply_parser import parse_ack
from transport.serial_interface import SerialLineIO
from ui.graphing import RuntimeMonitor, prompt_launch_gui, run_graph_tool


class EarlyStopOptimization(RuntimeError):
    pass


AXIS_NAMES = ("kp", "ki", "kd")


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


# Calculate per-trial metrics from the 5 repeated tests.
def compute_trial_metrics(
    per_test_powers: list[np.ndarray],
    per_test_meta: list[dict],
    desired_output: float,
):
    """Convert per-test power arrays into aggregate metrics used for scoring."""
    start_errors = []
    track_errors = []
    deviations = []
    max_errors = []
    strict_bad_rates = []
    oscillation_rates = []
    invalid_flags = []
    per_test_scores_unweighted = []

    for readings, meta in zip(per_test_powers, per_test_meta):
        if readings.size == 0:
            start_errors.append(999.0)
            track_errors.append(999.0)
            deviations.append(999.0)
            max_errors.append(999.0)
            strict_bad_rates.append(1.0)
            oscillation_rates.append(1.0)
            invalid_flags.append(1.0)
            per_test_scores_unweighted.append(999.0)
            continue

        start_power = float(readings[0])
        abs_error = np.abs(readings - desired_output)
        start_error = abs(start_power - desired_output)
        track_error = float(np.mean(abs_error))
        deviation = float(np.std(readings))
        max_error = float(np.max(abs_error))

        start_errors.append(start_error)
        track_errors.append(track_error)
        deviations.append(deviation)
        max_errors.append(max_error)
        strict_bad_rates.append(float(meta.get("strict_bad_rate", 1.0)))
        oscillation_rates.append(float(meta.get("oscillation_rate", 1.0)))
        invalid_flags.append(1.0 if bool(meta.get("invalid", False)) else 0.0)
        per_test_scores_unweighted.append(
            start_error + track_error + deviation + max_error + strict_bad_rates[-1] + oscillation_rates[-1]
        )

    return {
        "start_error": float(np.mean(start_errors)) if start_errors else 999.0,
        "track_error": float(np.mean(track_errors)) if track_errors else 999.0,
        "deviation": float(np.mean(deviations)) if deviations else 999.0,
        "max_error": float(np.mean(max_errors)) if max_errors else 999.0,
        "strict_bad_rate": float(np.mean(strict_bad_rates)) if strict_bad_rates else 1.0,
        "oscillation_rate": float(np.mean(oscillation_rates)) if oscillation_rates else 1.0,
        "invalid_ratio": float(np.mean(invalid_flags)) if invalid_flags else 1.0,
        "repeatability": float(np.std(per_test_scores_unweighted)) if per_test_scores_unweighted else 999.0,
    }


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
    return start_error + track_error + deviation + max_error + strict_bad_rate + oscillation_rate


# Convert weighted metrics into one scalar objective for PID candidate comparison.
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
    invalid_penalty: float,
    aborted: bool,
):
    """Combine metric values into one scalar score (lower is better)."""
    score = (
        w_start * metrics["start_error"]
        + w_track * metrics["track_error"]
        + w_dev * metrics["deviation"]
        + w_max * metrics["max_error"]
        + w_repeat * metrics["repeatability"]
        + w_strict * metrics["strict_bad_rate"]
        + w_osc * metrics["oscillation_rate"]
        + invalid_penalty * metrics["invalid_ratio"]
    )
    if aborted:
        score += 500.0
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
    if not safe_points:
        return {
            "ready": False,
            "reason": "no safe candidates yet",
            "unique_counts": (0, 0, 0),
            "spans": (0.0, 0.0, 0.0),
        }

    safe_arr = np.asarray(safe_points, dtype=float)
    unique_counts = tuple(int(len(np.unique(np.round(safe_arr[:, idx], 6)))) for idx in range(3))
    spans = tuple(float(np.max(safe_arr[:, idx]) - np.min(safe_arr[:, idx])) for idx in range(3))
    required_spans = (float(min_span_kp), float(min_span_ki), float(min_span_kd))

    if len(safe_points) < int(min_safe_candidates):
        return {
            "ready": False,
            "reason": f"only {len(safe_points)} safe candidates",
            "unique_counts": unique_counts,
            "spans": spans,
        }

    if len(good_points) < int(min_good_candidates):
        return {
            "ready": False,
            "reason": f"only {len(good_points)} good candidates",
            "unique_counts": unique_counts,
            "spans": spans,
        }

    for axis_name, unique_count in zip(AXIS_NAMES, unique_counts):
        if unique_count < int(min_points_per_axis):
            return {
                "ready": False,
                "reason": f"{axis_name} has only {unique_count} safe points",
                "unique_counts": unique_counts,
                "spans": spans,
            }

    for axis_name, span, required in zip(AXIS_NAMES, spans, required_spans):
        if span < required:
            return {
                "ready": False,
                "reason": f"{axis_name} span {span:.4f} < {required:.4f}",
                "unique_counts": unique_counts,
                "spans": spans,
            }

    return {
        "ready": True,
        "reason": "safe local region established",
        "unique_counts": unique_counts,
        "spans": spans,
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
) -> list[Real]:
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
    space: list[Real],
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
    """Build a short GUI checklist for BO readiness during warmup."""
    def mark(done: bool) -> str:
        return "[x]" if done else "[ ]"

    tail = f"Blocked by: {region_status['reason']}"
    if bool(region_status.get("ready")):
        tail = "Region ready for BO."
        if (
            warmup_trials_done is not None
            and warmup_trials_target is not None
            and warmup_trials_done < warmup_trials_target
        ):
            remaining = warmup_trials_target - warmup_trials_done
            tail = f"Region ready. Waiting for {remaining} more warmup trial(s) before BO starts."

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
    ap.add_argument("--w-start", type=float, default=1.0, help="Weight: start power error")
    ap.add_argument("--w-track", type=float, default=3.5, help="Weight: average tracking error")
    ap.add_argument("--w-dev", type=float, default=2.5, help="Weight: within-test deviation")
    ap.add_argument("--w-max", type=float, default=1.5, help="Weight: peak absolute error")
    ap.add_argument("--w-repeat", type=float, default=0.2, help="Weight: repeatability across 5 tests")
    ap.add_argument("--w-strict", type=float, default=6.0, help="Weight: settled +/-1% violations")
    ap.add_argument("--w-osc", type=float, default=3.0, help="Weight: post-settle oscillation")
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
        help="Minimum one-axis-at-a-time trials before Bayesian optimisation is allowed to start",
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
                f"Configured Bayesian trials: {n_trials} "
                "(candidate gathering warmup runs separately before BO starts)"
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
                monitor.set_phase("Phase: Gathering candidates")
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
            history = []
            power_rows = []
            trial_index = 0
            warmup_trial_count = 0
            bo_trial_count = 0
            baseline_score = None
            best_score_seen = float("inf")
            best_pid = None
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
            region_status = {
                "ready": False,
                "reason": "no safe candidates yet",
                "unique_counts": (0, 0, 0),
                "spans": (0.0, 0.0, 0.0),
            }
            if monitor is not None:
                warmup_target = max(1, int(args.coordinate_warmup_trials))
                monitor.set_warmup_counter(f"Warmup counter: 0/{warmup_target} completed")
                monitor.set_warmup_change("Warmup change: baseline trial using current hardware PID (no warmup delta)")
                monitor.set_previous_warmup_result("Previous warmup result: none yet")
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
                        warmup_trials_target=warmup_target,
                    )
                )

            def evaluate_candidate(
                kp: float,
                ki: float,
                kd: float,
                *,
                mode: str,
                used_axis: int | None = None,
            ) -> float:
                nonlocal trial_index, warmup_trial_count, bo_trial_count
                nonlocal baseline_score, best_score_seen, best_pid, last_applied
                nonlocal no_improve_count, step_kp, step_ki, step_kd, axis_index
                is_bayes_mode = mode == "bayes"
                display_phase_name = "BO" if is_bayes_mode else "Warmup"
                display_phase_index = (bo_trial_count + 1) if is_bayes_mode else (warmup_trial_count + 1)
                display_phase_total = n_trials if is_bayes_mode else None
                phase_repeats = max(1, int(args.bo_repeats if is_bayes_mode else args.warmup_repeats))
                log(
                    f"{display_phase_name} trial {display_phase_index}"
                    + (f"/{display_phase_total}" if display_phase_total is not None else "")
                    + f" (overall {trial_index + 1})"
                )
                if monitor is not None:
                    if is_bayes_mode:
                        monitor.set_phase("Phase: Bayesian Optimisation")
                        monitor.set_warmup_change("Warmup change: Bayesian search active")
                    else:
                        monitor.set_phase("Phase: Gathering candidates")
                    progress = f"{display_phase_name} trial {display_phase_index}"
                    if display_phase_total is not None:
                        progress = f"{progress}/{display_phase_total}"
                    monitor.set_progress(f"{progress} | configuring hardware | overall {trial_index + 1}")

                apply_pid_update = trial_index > 0
                if not apply_pid_update:
                    kp, ki, kd = 0.0, 0.0, 0.0
                    log("Baseline trial -> using current laser PID values without changing gains")
                elif mode == "bayes":
                    log(f"Bayesian candidate -> kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

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

                metrics = compute_trial_metrics(per_test_powers, per_test_meta, desired_output)
                score = score_controller(
                    metrics,
                    w_start=args.w_start,
                    w_track=args.w_track,
                    w_dev=args.w_dev,
                    w_max=args.w_max,
                    w_repeat=args.w_repeat,
                    w_strict=args.w_strict,
                    w_osc=args.w_osc,
                    invalid_penalty=args.invalid_penalty,
                    aborted=aborted,
                )

                prev_best_score = best_score_seen
                if baseline_score is None:
                    baseline_score = score
                best_score_seen = min(best_score_seen, score)

                improved = score < prev_best_score
                if improved:
                    best_pid = (used_kp, used_ki, used_kd)
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

                if apply_pid_update:
                    observed_points.append((used_kp, used_ki, used_kd))
                    observed_scores.append(score)
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
                        warmup_done = warmup_trial_count + 1
                        warmup_target = max(1, int(args.coordinate_warmup_trials))
                        remaining = max(0, warmup_target - warmup_done)
                        monitor.set_warmup_counter(
                            f"Warmup counter: {warmup_done}/{warmup_target} completed | {remaining} remaining"
                        )
                        if not is_bayes_mode:
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
                                warmup_trials_done=warmup_done,
                                warmup_trials_target=warmup_target,
                            )
                        )

                history.append(
                    (
                        used_kp,
                        used_ki,
                        used_kd,
                        score,
                        improve_vs_base_pct,
                        best_improve_vs_base_pct,
                        metrics["start_error"],
                        metrics["track_error"],
                        metrics["deviation"],
                        metrics["max_error"],
                        metrics["strict_bad_rate"],
                        metrics["oscillation_rate"],
                        metrics["invalid_ratio"],
                        metrics["repeatability"],
                        int(cancelled_candidate),
                        cancel_reason,
                        int(aborted),
                    )
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
                                trial_index + 1,
                                test_idx,
                                sample_idx,
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

                if (
                    args.retest_best_every > 0
                    and trial_index > 0
                    and best_pid is not None
                    and ((trial_index + 1) % args.retest_best_every == 0)
                ):
                    bkp, bki, bkd = best_pid
                    log(f"Re-testing best PID at interval -> kp={bkp:.4f}, ki={bki:.4f}, kd={bkd:.4f}")
                    _, by, _, baborted, _, btests, btimes, bmeta, _, _ = run_trial(
                        io,
                        bkp,
                        bki,
                        bkd,
                        desired_output=desired_output,
                        apply_pid_update=True,
                        repeats=max(1, int(args.bo_repeats)),
                        test_duration_s=test_duration_s,
                        startup_grace_s=args.startup_grace_s,
                        settled_window_samples=args.settled_window_samples,
                        duration=duration,
                        kp_max=args.kp_max,
                        ki_max=args.ki_max,
                        kd_max=args.kd_max,
                        monitor=monitor,
                        trial_index=trial_index + 1,
                        phase_name="BO Retest",
                        phase_trial_index=bo_trial_count if bo_trial_count > 0 else 1,
                        phase_trial_total=n_trials,
                        overall_trial_index=trial_index + 1,
                        repeat_cancel_osc_threshold=args.repeat_cancel_osc_threshold,
                        repeat_cancel_score_regression_pct=args.repeat_cancel_score_regression_pct,
                    )
                    bmetrics = compute_trial_metrics(btests, bmeta, desired_output)
                    bscore = score_controller(
                        bmetrics,
                        w_start=args.w_start,
                        w_track=args.w_track,
                        w_dev=args.w_dev,
                        w_max=args.w_max,
                        w_repeat=args.w_repeat,
                        w_strict=args.w_strict,
                        w_osc=args.w_osc,
                        invalid_penalty=args.invalid_penalty,
                        aborted=baborted,
                    )
                    log(
                        f"Best re-test -> score={bscore:.2f}, "
                        f"track_err={bmetrics['track_error']:.5f}, dev={bmetrics['deviation']:.5f}, "
                        f"repeat={bmetrics['repeatability']:.5f}, aborted={baborted}"
                    )
                    _ = by
                    _ = btimes

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
                    f"invalid={metrics['invalid_ratio']:.3f}, "
                    f"repeat={metrics['repeatability']:.5f}, "
                    f"cancelled={cancelled_candidate}, "
                    f"aborted={aborted}"
                )
                if cancelled_candidate:
                    log(f"Candidate repeats cancelled early: {cancel_reason}")

                trial_index += 1
                if is_bayes_mode:
                    bo_trial_count += 1
                else:
                    warmup_trial_count += 1
                if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
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
                        "Coordinate candidate -> "
                        f"base=({base_pid[0]:.4f},{base_pid[1]:.4f},{base_pid[2]:.4f}), "
                        f"axis={AXIS_NAMES[used_axis]}, delta={candidate_delta:+.4f}, "
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

                score = evaluate_candidate(kp, ki, kd, mode="coordinate", used_axis=used_axis)
                if used_axis is not None:
                    axis_index = (used_axis + 1) % 3
                return score

            log("Starting coordinate warmup")
            if monitor is not None:
                monitor.set_phase("Phase: Gathering candidates")
            try:
                warmup_target = max(1, int(args.coordinate_warmup_trials))
                max_warmup_trials = max(warmup_target, int(args.coordinate_warmup_trials) * 3, 12)
                while warmup_trial_count < max_warmup_trials and (
                    warmup_trial_count < warmup_target or not bool(region_status["ready"])
                ):
                    run_coordinate_trial()

                if bool(region_status["ready"]):
                    no_improve_count = 0
                    log(
                        f"Starting Bayesian optimisation after {warmup_trial_count} warmup trials; "
                        f"running {n_trials} BO trials"
                    )
                    if monitor is not None:
                        monitor.set_phase("Phase: Bayesian Optimisation")
                        monitor.set_warmup_counter("")
                        monitor.set_readiness("BO readiness:\n[x] Ready. Bayesian optimisation started.")
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
                        lambda x: evaluate_candidate(float(x[0]), float(x[1]), float(x[2]), mode="bayes"),
                        bayes_space,
                        n_calls=n_trials,
                        n_initial_points=min(4, max(1, n_trials)),
                        acq_func="EI",
                        random_state=42,
                        x0=seed_points if seed_points else None,
                        y0=seed_scores if seed_scores else None,
                    )
                else:
                    log(
                        "Bayesian optimisation skipped: "
                        f"no viable region found after {warmup_trial_count} warmup trials "
                        f"({region_status['reason']})"
                    )
                    if monitor is not None:
                        monitor.set_phase("Phase: Gathering candidates")
                        monitor.set_progress(
                            f"Warmup stopped after {warmup_trial_count} trials | {region_status['reason']}"
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
                                warmup_trials_done=warmup_trial_count,
                                warmup_trials_target=warmup_target,
                            )
                        )
            except EarlyStopOptimization as e:
                log(f"Early stop: {e}")

            if best_pid is not None:
                best_kp, best_ki, best_kd = best_pid
            elif last_applied is not None:
                best_kp, best_ki, best_kd = last_applied
            else:
                best_kp, best_ki, best_kd = 0.0, 0.0, 0.0
            log("Tuning complete")
            if monitor is not None:
                monitor.set_phase("Phase: Run complete")
            log(f"BEST kp={best_kp:.6f}, ki={best_ki:.6f}, kd={best_kd:.6f}")
            if best_score_seen < float("inf"):
                log(f"Best score={best_score_seen:.3f}")

            with open("tuning_history.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "kp",
                        "ki",
                        "kd",
                        "score",
                        "improve_vs_baseline_pct",
                        "best_improve_vs_baseline_pct",
                        "start_error",
                        "track_error",
                        "deviation",
                        "max_error",
                        "strict_bad_rate",
                        "oscillation_rate",
                        "invalid_ratio",
                        "repeatability",
                        "cancelled_repeats",
                        "cancel_reason",
                        "aborted",
                    ]
                )
                writer.writerows(history)
            log("Saved tuning_history.csv")

            with open("tuning_power_readings.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "trial_index",
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

"""Candidate execution helpers for repeated hardware trials."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import time

import numpy as np

import laser_command_ids as CMD
from pipeline.data_collector import StartupTelemetryTimeoutError, collect_trial_data
from protocol.reply_parser import parse_ack
from transport.serial_interface import SerialLineIO
from tuning.metrics import score_single_repeat
from ui.graphing import RuntimeMonitor


# Keep trial-runner logging timestamped and consistent with the main script.
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# Format progress text once here so the monitor sees the same wording from
# every phase.
def _candidate_progress_message(
    *,
    phase_name: str | None,
    phase_trial_index: int | None,
    phase_trial_total: int | None,
    overall_trial_index: int | None,
    trial_index: int | None,
    suffix: str,
) -> str:
    if phase_name is not None and phase_trial_index is not None:
        progress = f"{phase_name} trial {phase_trial_index}"
        if phase_trial_total is not None:
            progress = f"{progress}/{phase_trial_total}"
        if overall_trial_index is not None:
            progress = f"{progress} | overall {overall_trial_index}"
        return f"{progress} | {suffix}"
    if trial_index is not None:
        return f"Trial {trial_index} | {suffix}"
    return suffix


# Categorise repeat failures so later two-strike cancellation logic can reason
# about repeated bad behaviour rather than one-off noise.
def _repeat_failure_categories(
    *,
    meta: dict,
    repeat_score: float,
    previous_valid_scores: list[float],
    repeat_cancel_osc_threshold: float,
    repeat_cancel_score_regression_pct: float,
) -> tuple[set[str], float]:
    categories: set[str] = set()
    regression_pct = 0.0
    if bool(meta.get("invalid", False)):
        categories.add("invalid_repeat")
    if bool(meta.get("start_skewed", False)):
        categories.add("start_skew")
    if float(meta.get("oscillation_rate", 0.0)) >= float(repeat_cancel_osc_threshold):
        categories.add("high_oscillation")
    if len(previous_valid_scores) >= 2:
        reference = float(np.median(np.asarray(previous_valid_scores, dtype=float)))
        if reference > 1e-9:
            regression_pct = 100.0 * (float(repeat_score) - reference) / reference
            if regression_pct >= float(repeat_cancel_score_regression_pct):
                categories.add("severe_score_regression")
    return categories, float(regression_pct)


# Collapse strike history into one readable cancellation reason.
def _summarize_strikes(strike_history: dict[str, list[int]]) -> str:
    for category, repeats in strike_history.items():
        if len(repeats) >= 2:
            repeats_str = " and ".join(str(rep) for rep in repeats[:2])
            return f"two-strike cancellation: {category} occurred in repeats {repeats_str}"
    return ""


# Orchestrate one full candidate evaluation on real hardware:
# configure debug output, optionally apply PID values, run repeated START/STOP
# cycles, collect telemetry, and decide whether the candidate should be
# cancelled early.
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
    startup_telemetry_timeout_s: float = 5.0,
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
    best_pid: tuple[float, float, float] | None = None,
    repeat_cancel_osc_threshold: float = 0.35,
    repeat_cancel_score_regression_pct: float = 8.0,
):
    """Run one PID candidate through repeated laser tests and collect telemetry."""
    _ = duration
    log(f"Starting trial: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

    kp = float(np.clip(kp, 0.0, kp_max))
    ki = float(np.clip(ki, 0.0, ki_max))
    kd = float(np.clip(kd, 0.0, kd_max))

    try:
        # Enable the B0 telemetry/debug stream before any repeats begin.
        io.write_command_expect_ok_ack("000B", command_id_hex2=CMD.SET_DEBUG, timeout=2.0)
        log("SET_DEBUG acknowledged with *00")
    except Exception as exc:
        raise RuntimeError(f"SET_DEBUG did not receive success ACK '*00': {exc}") from exc

    try:
        current_pid = io.get_pid_values(timeout=2.0)
        log(
            "Current PID values: "
            f"PW Kp={current_pid['pw_kp']:.4f}, "
            f"Ki={current_pid['pw_ki']:.4f}, "
            f"Kd={current_pid['pw_kd']:.4f}"
        )
    except Exception as exc:
        log(f"Warning: Could not read current PID values: {exc}. Using defaults for PP parameters.")
        current_pid = None

    sample_interval_s = None
    if current_pid is not None:
        # Some firmware reports the sample interval in milliseconds, others in
        # seconds, so normalise it here once.
        raw_si = float(current_pid.get("sample_interval", 0.0))
        if raw_si > 0:
            sample_interval_s = raw_si / 1000.0 if raw_si > 1.0 else raw_si
            log(f"Using telemetry sample interval: {sample_interval_s:.6f}s")

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
        # Baseline trials keep the controller's existing PID, so show that in
        # the monitor instead of the placeholder zeros used in the caller.
        display_kp = float(current_pid["pw_kp"])
        display_ki = float(current_pid["pw_ki"])
        display_kd = float(current_pid["pw_kd"])
    if monitor is not None:
        monitor.set_target(desired_output)
        monitor.set_pid_values(
            display_kp,
            display_ki,
            display_kd,
            best_pid=best_pid,
        )
        monitor.set_progress(
            _candidate_progress_message(
                phase_name=phase_name,
                phase_trial_index=phase_trial_index,
                phase_trial_total=phase_trial_total,
                overall_trial_index=overall_trial_index,
                trial_index=trial_index,
                suffix="configuring hardware",
            )
        )

    t_vals, y_vals, u_vals, status_vals = [], [], [], []
    per_test_powers: list[np.ndarray] = []
    per_test_times: list[np.ndarray] = []
    per_test_meta: list[dict] = []
    repeat_scores: list[float] = []
    cancelled_candidate = False
    cancel_reason = ""
    invalid_repeat_count = 0
    min_valid_repeats = max(2, int(np.ceil(max(1, repeats) / 2.0)))
    strike_history: dict[str, list[int]] = defaultdict(list)
    tolerated_bad_repeat_logged = False

    # Bring the controller into RUN and open the shutter once for the whole
    # candidate. Individual repeats then use START/STOP inside that window.
    io.write_command_expect_ok_ack("", command_id_hex2=CMD.RUN, timeout=2.0)
    io.write_command_expect_ok_ack("1", command_id_hex2=CMD.SHUTTER_CONTROL, timeout=2.0)
    log("Shutter opened; waiting 1.0s before sending the next command")
    time.sleep(1.0)

    try:
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
                accepted_codes=("00", "08", "80"),
            )
            log(f"Test {rep + 1}/{repeats}: START acknowledged")

            test_meta = {
                # Trial metadata carries repeat-level decisions back to the
                # scorer and bootstrap logic.
                "invalid": False,
                "reason": "",
                "settled": False,
                "strict_bad_rate": 1.0,
                "oscillation_rate": 1.0,
                "stopped_early": False,
                "start_skewed": False,
                "start_skew_error": 0.0,
                "failure_categories": [],
                "score_regression_pct": 0.0,
                "unsafe_repeat": False,
                "cancellation_decision": "",
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
                    # The first sample is used for the startup skew check before
                    # the repeat has had any chance to settle.
                    first_seen = True
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

                if (not test_meta["settled"]) and len(recent_within_5pct) == settled_window_samples and all(recent_within_5pct):
                    test_meta["settled"] = True

                if test_meta["settled"]:
                    # Once settled, leaving the +/-5% band invalidates the
                    # repeat immediately and sends STOP early.
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

            def on_waiting_for_first_sample() -> None:
                # Surface the "armed but not yet receiving telemetry" state in
                # both logs and the GUI.
                log(
                    f"Test {rep + 1}/{repeats}: waiting for first telemetry "
                    f"(timeout {startup_telemetry_timeout_s:.2f}s)"
                )
                if monitor is not None:
                    monitor.set_progress(
                        _candidate_progress_message(
                            phase_name=phase_name,
                            phase_trial_index=phase_trial_index,
                            phase_trial_total=phase_trial_total,
                            overall_trial_index=overall_trial_index,
                            trial_index=trial_index,
                            suffix=f"test {rep + 1}/{repeats} START acknowledged, waiting for first telemetry",
                        )
                    )

            def on_first_sample(startup_delay_s: float, mapped: dict) -> None:
                # The scored window starts from this first parsed telemetry
                # packet, not from the START ACK.
                log(
                    f"Test {rep + 1}/{repeats}: first valid telemetry received after "
                    f"{startup_delay_s:.3f}s (status={mapped.get('status', 'RUNNING')})"
                )
                if monitor is not None:
                    monitor.set_progress(
                        _candidate_progress_message(
                            phase_name=phase_name,
                            phase_trial_index=phase_trial_index,
                            phase_trial_total=phase_trial_total,
                            overall_trial_index=overall_trial_index,
                            trial_index=trial_index,
                            suffix=f"test {rep + 1}/{repeats} active",
                        )
                    )

            try:
                rt, ry, ru, rs = collect_trial_data(
                    io,
                    line_timeout=0.5,
                    sample_interval_s=sample_interval_s,
                    duration_s=test_duration_s,
                    wait_for_first_sample_timeout_s=startup_telemetry_timeout_s,
                    stop_on_done=False,
                    on_sample=on_sample,
                    on_waiting_for_first_sample=on_waiting_for_first_sample,
                    on_first_sample=on_first_sample,
                )
            except StartupTelemetryTimeoutError as exc:
                # Treat missing startup telemetry as an invalid repeat with its
                # own explicit reason.
                rt, ry, ru, rs = [], [], [], []
                test_meta["invalid"] = True
                test_meta["reason"] = str(exc)
                log(
                    f"Test {rep + 1}/{repeats}: startup telemetry wait timed out after "
                    f"{startup_telemetry_timeout_s:.2f}s"
                )
                if monitor is not None:
                    monitor.set_progress(
                        _candidate_progress_message(
                            phase_name=phase_name,
                            phase_trial_index=phase_trial_index,
                            phase_trial_total=phase_trial_total,
                            overall_trial_index=overall_trial_index,
                            trial_index=trial_index,
                            suffix=f"test {rep + 1}/{repeats} invalid: no telemetry received after START",
                        )
                    )

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
            if "ABORT" in rs:
                test_meta["unsafe_repeat"] = True
                if not test_meta["reason"]:
                    test_meta["reason"] = "hardware safety abort"
            if not test_meta["settled"] and not test_meta["invalid"] and not test_meta["reason"]:
                test_meta["reason"] = "did not settle"

            repeat_score = score_single_repeat(per_test_powers[-1], test_meta, desired_output)
            previous_valid_scores = [
                score
                for score, meta in zip(repeat_scores, per_test_meta)
                if not bool(meta.get("invalid", False))
            ]
            failure_categories, regression_pct = _repeat_failure_categories(
                meta=test_meta,
                repeat_score=repeat_score,
                previous_valid_scores=previous_valid_scores,
                repeat_cancel_osc_threshold=repeat_cancel_osc_threshold,
                repeat_cancel_score_regression_pct=repeat_cancel_score_regression_pct,
            )
            test_meta["failure_categories"] = sorted(failure_categories)
            test_meta["score_regression_pct"] = float(regression_pct)

            per_test_meta.append(test_meta)
            repeat_scores.append(repeat_score)

            if ry:
                # Log a quick summary for the operator without dumping the full
                # trace.
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
                invalid_repeat_count += 1
                log(f"Test {rep + 1}/{repeats} invalid: {test_meta['reason']}")
            elif test_meta["reason"]:
                log(f"Test {rep + 1}/{repeats} note: {test_meta['reason']}")

            for category in test_meta["failure_categories"]:
                strike_history[category].append(rep + 1)

            if test_meta["unsafe_repeat"]:
                # Safety aborts always cancel the remaining repeats.
                cancelled_candidate = True
                cancel_reason = f"unsafe cancellation: {test_meta['reason'] or 'hardware safety abort'}"
                test_meta["cancellation_decision"] = cancel_reason
            else:
                strike_cancel_reason = _summarize_strikes(strike_history)
                if strike_cancel_reason:
                    cancelled_candidate = True
                    cancel_reason = strike_cancel_reason
                    test_meta["cancellation_decision"] = cancel_reason

            remaining_repeats = repeats - (rep + 1)
            valid_repeats = len(per_test_meta) - invalid_repeat_count
            if (
                not cancelled_candidate
                and valid_repeats + remaining_repeats < min_valid_repeats
            ):
                # Stop early once it becomes mathematically impossible to finish
                # with enough valid repeats.
                cancelled_candidate = True
                cancel_reason = (
                    "unrecoverable cancellation: "
                    f"only {valid_repeats} valid repeats with {remaining_repeats} remaining "
                    f"(need {min_valid_repeats} valid repeats)"
                )
                test_meta["cancellation_decision"] = cancel_reason

            if test_meta["failure_categories"] and not cancelled_candidate and not tolerated_bad_repeat_logged:
                categories = ", ".join(test_meta["failure_categories"])
                log(f"no cancellation: isolated bad repeat tolerated ({categories} on repeat {rep + 1})")
                tolerated_bad_repeat_logged = True

            if not test_meta["stopped_early"]:
                io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
                log(f"Test {rep + 1}/{repeats}: STOP")
                if monitor is not None:
                    monitor.set_progress(
                        _candidate_progress_message(
                            phase_name=phase_name,
                            phase_trial_index=phase_trial_index,
                            phase_trial_total=phase_trial_total,
                            overall_trial_index=overall_trial_index,
                            trial_index=trial_index,
                            suffix=f"test {rep + 1}/{repeats} stopped",
                        )
                    )
            else:
                log(f"Test {rep + 1}/{repeats}: STOP (already sent on invalid condition)")

            if cancelled_candidate:
                log(f"Cancelling remaining repeats for this PID candidate: {cancel_reason}")
                break
    finally:
        # Always try to leave the hardware in a safe idle state, even if a
        # repeat raised midway through collection.
        try:
            io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
        except Exception:
            pass
        io.write_command_expect_ok_ack("0", command_id_hex2=CMD.SHUTTER_CONTROL, timeout=2.0)
        io.write_command_expect_ok_ack("", command_id_hex2=CMD.STANDBY, timeout=2.0)
        log("End of PID set: shutter closed, standby set")
        if monitor is not None:
            monitor.set_progress(
                _candidate_progress_message(
                    phase_name=phase_name,
                    phase_trial_index=phase_trial_index,
                    phase_trial_total=phase_trial_total,
                    overall_trial_index=overall_trial_index,
                    trial_index=trial_index,
                    suffix="shutter closed, standby set",
                )
            )

    aborted = any(s == "ABORT" for s in status_vals)
    start_skew_count = sum(1 for meta in per_test_meta if bool(meta.get("start_skewed", False)))
    start_skew_threshold = max(2, (len(per_test_meta) // 2) + 1) if per_test_meta else 2
    if start_skew_count >= start_skew_threshold:
        summary_reason = f"start skew exceeded +/-30% in {start_skew_count}/{len(per_test_meta)} repeats"
        for meta in per_test_meta:
            if bool(meta.get("start_skewed", False)) and not bool(meta.get("invalid", False)):
                meta["invalid"] = True
                meta["reason"] = summary_reason
                categories = set(meta.get("failure_categories", []))
                categories.add("start_skew")
                meta["failure_categories"] = sorted(categories)
        if repeat_scores:
            repeat_scores = [
                999.0 if bool(meta.get("start_skewed", False)) else score
                for meta, score in zip(per_test_meta, repeat_scores)
            ]
        log(f"Candidate start-skew rule triggered: {summary_reason}")
    if aborted:
        log("Warning: Trial aborted due to safety condition")
    if cancelled_candidate:
        log(f"Trial ended early: {cancel_reason}")

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

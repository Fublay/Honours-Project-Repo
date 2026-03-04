"""Main application entry point for laser PID tuning.

This script provides:
- startup UI (GUI/CLI) for running actions
- serial command orchestration for each laser test
- trial scoring and Bayesian optimization
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


class EarlyStopOptimization(RuntimeError):
    pass


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


def prompt_launch_gui(default_goal: float, default_trials: int, default_test_duration_s: float):
    """Show the startup GUI and return selected action + field values.

    Return shape:
        {"action": "start|reset|graph|quit", "goal": float, "trials": int, "test_duration_s": float}
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return None

    result = {
        "action": None,
        "goal": float(default_goal),
        "trials": int(default_trials),
        "test_duration_s": float(default_test_duration_s),
    }

    try:
        root = tk.Tk()
    except Exception:
        return None

    root.title("PID Tuner")
    root.resizable(False, False)

    frame = tk.Frame(root, padx=12, pady=12)
    frame.grid(row=0, column=0, sticky="nsew")

    tk.Label(frame, text="Goal Power Output").grid(row=0, column=0, sticky="w")
    goal_var = tk.StringVar(value=f"{float(default_goal):.4f}")
    goal_entry = tk.Entry(frame, textvariable=goal_var, width=16)
    goal_entry.grid(row=1, column=0, sticky="w", pady=(2, 8))

    tk.Label(frame, text="Number of Trials").grid(row=2, column=0, sticky="w")
    trials_var = tk.StringVar(value=str(int(default_trials)))
    trials_entry = tk.Entry(frame, textvariable=trials_var, width=16)
    trials_entry.grid(row=3, column=0, sticky="w", pady=(2, 10))

    tk.Label(frame, text="Test Duration (s)").grid(row=4, column=0, sticky="w")
    duration_var = tk.StringVar(value=f"{float(default_test_duration_s):.1f}")
    duration_entry = tk.Entry(frame, textvariable=duration_var, width=16)
    duration_entry.grid(row=5, column=0, sticky="w", pady=(2, 10))

    def parse_fields() -> bool:
        try:
            goal = float(goal_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Goal power output must be numeric.")
            return False
        try:
            trials = int(trials_var.get().strip())
            if trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of trials must be an integer >= 1.")
            return False
        try:
            test_duration_s = float(duration_var.get().strip())
            if test_duration_s <= 0.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Test duration must be a number > 0.")
            return False
        result["goal"] = goal
        result["trials"] = trials
        result["test_duration_s"] = test_duration_s
        return True

    def on_start():
        if not parse_fields():
            return
        result["action"] = "start"
        root.destroy()

    def on_reset():
        result["action"] = "reset"
        root.destroy()

    def on_quit():
        result["action"] = "quit"
        root.destroy()

    def on_graph():
        result["action"] = "graph"
        root.destroy()

    btn_row = tk.Frame(frame)
    btn_row.grid(row=6, column=0, sticky="w")
    tk.Button(btn_row, text="Start Test", width=12, command=on_start).grid(row=0, column=0, padx=(0, 6))
    tk.Button(btn_row, text="Reset Defaults", width=12, command=on_reset).grid(row=0, column=1, padx=(0, 6))
    tk.Button(btn_row, text="Graph Power", width=11, command=on_graph).grid(row=0, column=2, padx=(0, 6))
    tk.Button(btn_row, text="Quit", width=8, command=on_quit).grid(row=0, column=3)

    root.protocol("WM_DELETE_WINDOW", on_quit)
    goal_entry.focus_set()
    root.mainloop()

    if result["action"] is None:
        result["action"] = "quit"
    return result


def load_power_series(csv_path: str):
    """Load saved power CSV rows and group them by (trial_index, test_index)."""
    series_map = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                trial = int(row["trial_index"])
                test = int(row["test_index"])
                t_s = float(row["time_s"])
                power = float(row["current_power"])
                goal = float(row["desired_output"])
            except (ValueError, KeyError):
                continue
            key = (trial, test)
            bucket = series_map.setdefault(key, {"time_s": [], "power": [], "goal": goal})
            bucket["time_s"].append(t_s)
            bucket["power"].append(power)
            bucket["goal"] = goal

    final = {}
    for key, payload in series_map.items():
        final[key] = {
            "time_s": np.array(payload["time_s"], dtype=float),
            "power": np.array(payload["power"], dtype=float),
            "goal": float(payload["goal"]),
        }
    return final


def plot_power_tests(csv_path: str, first_key: tuple[int, int], second_key: tuple[int, int] | None = None):
    """Plot one or two selected test traces from CSV data."""
    import matplotlib.pyplot as plt

    series = load_power_series(csv_path)
    if first_key not in series:
        raise RuntimeError(f"Test not found: trial={first_key[0]}, test={first_key[1]}")
    if second_key is not None and second_key not in series:
        raise RuntimeError(f"Test not found: trial={second_key[0]}, test={second_key[1]}")

    fig, ax = plt.subplots(figsize=(10, 5))
    first = series[first_key]
    ax.plot(first["time_s"], first["power"], label=f"Trial {first_key[0]} Test {first_key[1]}")
    ax.axhline(first["goal"], linestyle="--", linewidth=1.2, alpha=0.8, label=f"Goal {first['goal']:.2f}")

    if second_key is not None:
        second = series[second_key]
        ax.plot(second["time_s"], second["power"], label=f"Trial {second_key[0]} Test {second_key[1]}")
        if abs(second["goal"] - first["goal"]) > 1e-9:
            ax.axhline(
                second["goal"],
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                label=f"Goal 2 {second['goal']:.2f}",
            )

    ax.set_title("Power Readings")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current Power")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


def plot_power_tests_interactive(csv_path: str):
    """Open interactive graph window with checkbox-based test visibility control."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons

    series = load_power_series(csv_path)
    if not series:
        raise RuntimeError(f"No valid readings found in {csv_path}")

    keys = sorted(series.keys())

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.4, 4.6])
    ax_checks = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    lines_by_label = {}
    labels = []
    goals_plotted = set()
    for key in keys:
        trial, test = key
        label = f"T{trial}-Test{test}"
        payload = series[key]
        line, = ax_plot.plot(payload["time_s"], payload["power"], label=label, visible=False)
        lines_by_label[label] = line
        labels.append(label)
        goal_val = float(payload["goal"])
        if goal_val not in goals_plotted:
            ax_plot.axhline(goal_val, linestyle="--", linewidth=1.0, alpha=0.5, label=f"Goal {goal_val:.2f}")
            goals_plotted.add(goal_val)

    check = CheckButtons(ax_checks, labels, [False] * len(labels))

    def visible_lines():
        return [line for line in lines_by_label.values() if line.get_visible()]

    def apply_default_view():
        # Keep the baseline display behavior with a zero-based Y axis.
        ax_plot.relim()
        ax_plot.autoscale_view(scalex=True, scaley=True)
        y_top = ax_plot.get_ylim()[1]
        if not np.isfinite(y_top) or y_top <= 0.0:
            y_top = 1.0
        ax_plot.set_ylim(bottom=0.0, top=y_top)

    def apply_zoomed_view():
        # Zoom around currently visible traces; if none are visible, use all traces.
        candidates = visible_lines()
        if not candidates:
            candidates = list(lines_by_label.values())
        if not candidates:
            return

        ymins = []
        ymaxs = []
        for line in candidates:
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size == 0:
                continue
            ymins.append(float(np.min(y)))
            ymaxs.append(float(np.max(y)))
        if not ymins:
            return

        y_min = min(ymins)
        y_max = max(ymaxs)
        span = max(y_max - y_min, 1e-6)
        pad = max(0.5, span * 0.08)
        ax_plot.set_ylim(y_min - pad, y_max + pad)

    def on_clicked(label):
        line = lines_by_label[label]
        line.set_visible(not line.get_visible())
        ax_plot.legend(loc="best")
        fig.canvas.draw_idle()

    check.on_clicked(on_clicked)

    # Add quick controls to toggle all traces at once.
    ax_all_on = fig.add_axes([0.08, 0.08, 0.10, 0.06])
    ax_all_off = fig.add_axes([0.19, 0.08, 0.10, 0.06])
    ax_default_view = fig.add_axes([0.30, 0.08, 0.12, 0.06])
    ax_zoom_view = fig.add_axes([0.43, 0.08, 0.12, 0.06])
    btn_all_on = Button(ax_all_on, "All On")
    btn_all_off = Button(ax_all_off, "All Off")
    btn_default_view = Button(ax_default_view, "Default View")
    btn_zoom_view = Button(ax_zoom_view, "Zoomed View")

    def set_all(target_visible: bool):
        statuses = list(check.get_status())
        for idx, is_on in enumerate(statuses):
            if bool(is_on) != bool(target_visible):
                check.set_active(idx)
        ax_plot.legend(loc="best")
        fig.canvas.draw_idle()

    btn_all_on.on_clicked(lambda _evt: set_all(True))
    btn_all_off.on_clicked(lambda _evt: set_all(False))
    btn_default_view.on_clicked(lambda _evt: (apply_default_view(), fig.canvas.draw_idle()))
    btn_zoom_view.on_clicked(lambda _evt: (apply_zoomed_view(), fig.canvas.draw_idle()))

    ax_checks.set_title("Select Tests")
    ax_plot.set_title("Power Readings")
    ax_plot.set_xlabel("Time (s)")
    ax_plot.set_ylabel("Current Power")
    apply_default_view()
    ax_plot.grid(True, alpha=0.25)
    ax_plot.legend(loc="best")
    fig.tight_layout()
    plt.show()


def run_graph_tool(csv_path: str, prefer_gui: bool = True):
    """Run graph view workflow and surface readable errors for missing dependencies."""
    try:
        series = load_power_series(csv_path)
    except FileNotFoundError:
        raise RuntimeError(f"Power readings file not found: {csv_path}")

    if not series:
        raise RuntimeError(f"No valid readings found in {csv_path}")

    _ = series
    _ = prefer_gui
    try:
        plot_power_tests_interactive(csv_path)
    except ModuleNotFoundError as e:
        if e.name == "matplotlib":
            raise RuntimeError(
                "Graphing requires matplotlib in your active environment. "
                "Install with: pip install matplotlib"
            ) from e
        raise


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

    # Prepare output arrays for all repeats under this PID candidate.
    t_vals, y_vals, u_vals, status_vals = [], [], [], []
    per_test_powers: list[np.ndarray] = []
    per_test_times: list[np.ndarray] = []
    per_test_meta: list[dict] = []

    # Put the laser into run mode and open shutter before individual test starts.
    io.write_command_expect_ok_ack("", command_id_hex2=CMD.RUN, timeout=2.0)
    io.write_command_expect_ok_ack("1", command_id_hex2=CMD.SHUTTER_CONTROL, timeout=2.0)

    try:
        # Repeat the same PID candidate several times to reduce one-off noise.
        for rep in range(repeats):
            log(f"Test {rep + 1}/{repeats}: START")
            io.write_command_expect_ok_ack(
                "",
                command_id_hex2=CMD.START,
                timeout=2.0,
                accepted_codes=("00", "80"),
            )

            test_meta = {
                "invalid": False,
                "reason": "",
                "settled": False,
                "strict_bad_rate": 1.0,
                "oscillation_rate": 1.0,
                "stopped_early": False,
            }
            recent_within_5pct = []
            strict_bad_count = 0
            strict_total = 0
            prev_sign = 0
            sign_flips = 0
            sign_total = 0
            first_seen = False

            base = max(abs(desired_output), 1e-6)
            limit_30 = 0.30 * base
            limit_5 = 0.05 * base
            limit_1 = 0.01 * base

            def on_sample(t_val, mapped) -> bool:
                nonlocal strict_bad_count, strict_total, prev_sign, sign_flips, sign_total, first_seen
                y_val = float(mapped["process_value"])
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
                        test_meta["invalid"] = True
                        test_meta["reason"] = (
                            f"first reading out of +/-30% "
                            f"(initial={first_power:.4f}, target={desired_output:.4f})"
                        )
                        try:
                            io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
                            test_meta["stopped_early"] = True
                        except Exception as stop_err:
                            log(f"Warning: failed to send immediate STOP on invalid test: {stop_err}")
                        return True

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

                    sign = 1 if err > 0 else (-1 if err < 0 else 0)
                    if prev_sign != 0 and sign != 0:
                        sign_total += 1
                        if sign != prev_sign:
                            sign_flips += 1
                    if sign != 0:
                        prev_sign = sign

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

            if sign_total > 0:
                test_meta["oscillation_rate"] = sign_flips / sign_total
            else:
                test_meta["oscillation_rate"] = 1.0 if test_meta["settled"] else 0.0

            if not ry and not test_meta["invalid"]:
                test_meta["invalid"] = True
                test_meta["reason"] = "no samples collected"
            if not test_meta["settled"] and not test_meta["invalid"]:
                # Did not settle within test window: keep valid, but penalize in metrics.
                test_meta["reason"] = "did not settle"

            per_test_meta.append(test_meta)

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

            if not test_meta["stopped_early"]:
                io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
                log(f"Test {rep + 1}/{repeats}: STOP")
            else:
                log(f"Test {rep + 1}/{repeats}: STOP (already sent on invalid condition)")
    finally:
        # Always leave hardware in a safe idle state at end of a candidate.
        try:
            io.write_command_expect_ok_ack("", command_id_hex2=CMD.STOP, timeout=2.0)
        except Exception:
            pass
        io.write_command_expect_ok_ack("0", command_id_hex2=CMD.SHUTTER_CONTROL, timeout=2.0)
        io.write_command_expect_ok_ack("", command_id_hex2=CMD.STANDBY, timeout=2.0)
        log("End of PID set: shutter closed, standby set")

    aborted = any(s == "ABORT" for s in status_vals)
    if aborted:
        log("Warning: Trial aborted due to safety condition")

    return (
        np.array(t_vals),
        np.array(y_vals),
        np.array(u_vals),
        aborted,
        current_pid,
        per_test_powers,
        per_test_times,
        per_test_meta,
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


# Convert weighted metrics into one scalar objective for Bayesian optimization.
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


def main():
    """Parse inputs, run selected action, and manage full tuning workflow."""
    import argparse

    # Runtime arguments for serial connection, optimizer bounds, and target output.
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Serial port (e.g. /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--iters", type=int, default=20, help="Number of optimisation iterations")
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
    args = ap.parse_args()

    action = None
    desired_output = None
    n_trials = None
    test_duration_s = None
    if not args.no_gui:
        ui = prompt_launch_gui(args.desired_output, args.iters, args.test_duration_s)
        if ui is not None:
            action = ui["action"]
            desired_output = float(ui["goal"])
            n_trials = int(ui["trials"])
            test_duration_s = float(ui["test_duration_s"])
        else:
            log("GUI unavailable; falling back to console prompts.")

    if action is None:
        action = prompt_launch_action()
        if action == "start":
            desired_output = prompt_goal_power_output(args.desired_output)
            n_trials = prompt_trial_count(args.iters)

    if action == "quit":
        log("Exiting on user request.")
        return
    if action == "graph":
        try:
            run_graph_tool(args.power_csv, prefer_gui=(not args.no_gui))
        except RuntimeError as e:
            log(f"Graph tool error: {e}")
        log("Graph tool closed.")
        return

    # Open serial transport once and reuse it across the optimization run.
    log("Opening serial port")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    io = SerialLineIO(
        ser,
        log_fn=log,
        log_data_lines=args.log_data,
        data_log_every=args.log_data_every,
    )

    if action == "reset":
        log("Resetting PID values to defaults")
        reset_pid_defaults(io)
        log("Defaults restored successfully.")
        return

    if desired_output is None:
        desired_output = prompt_goal_power_output(args.desired_output)
    log(f"Goal power output set to {desired_output:.4f}")
    if n_trials is None:
        n_trials = prompt_trial_count(args.iters)
    log(f"Configured trials: {n_trials} (total laser runs: {n_trials * 5})")
    if test_duration_s is None:
        test_duration_s = float(args.test_duration_s)
    log(f"Configured per-test duration: {test_duration_s:.2f}s")

    duration = 15.0
    space = [
        Real(0.0, args.kp_max, name="kp"),
        Real(0.0, args.ki_max, name="ki"),
        Real(0.0, args.kd_max, name="kd"),
    ]

    # Keep a trial-by-trial record for later review.
    history = []
    power_rows = []
    trial_index = 0
    baseline_score = None
    best_score_seen = float("inf")
    best_pid = None
    last_applied = None
    no_improve_count = 0
    refine_mode = False
    recovery_mode = False
    recovery_exact_next = True
    stagnation_axis = 0
    stagnation_sign = 1.0
    step_kp = float(args.max_step_kp)
    step_ki = float(args.max_step_ki)
    step_kd = float(args.max_step_kd)
    min_step_kp = max(0.01, step_kp * 0.1)
    min_step_ki = max(0.01, step_ki * 0.1)
    min_step_kd = max(0.005, step_kd * 0.1)

    def objective(x):
        nonlocal trial_index, baseline_score, best_score_seen, best_pid, last_applied
        nonlocal no_improve_count, refine_mode
        nonlocal recovery_mode, recovery_exact_next
        nonlocal stagnation_axis, stagnation_sign
        nonlocal step_kp, step_ki, step_kd
        kp, ki, kd = x
        log(f"Trial {trial_index + 1}/{n_trials}")

        best_improve_pct_so_far = 0.0
        if baseline_score and baseline_score > 0 and best_score_seen < float("inf"):
            best_improve_pct_so_far = 100.0 * (baseline_score - best_score_seen) / baseline_score

        # Once we are clearly improving, keep search near best-known PID values.
        if best_pid is not None and best_improve_pct_so_far >= args.refine_activate_improve_pct:
            bkp, bki, bkd = best_pid
            kp_raw, ki_raw, kd_raw = kp, ki, kd
            kp = float(np.clip(kp, bkp - args.refine_radius_kp, bkp + args.refine_radius_kp))
            ki = float(np.clip(ki, bki - args.refine_radius_ki, bki + args.refine_radius_ki))
            kd = float(np.clip(kd, bkd - args.refine_radius_kd, bkd + args.refine_radius_kd))
            if not refine_mode:
                refine_mode = True
                log(
                    "Refinement mode active around best PID: "
                    f"radius=({args.refine_radius_kp:.3f},{args.refine_radius_ki:.3f},{args.refine_radius_kd:.3f})"
                )
            if (kp, ki, kd) != (kp_raw, ki_raw, kd_raw):
                log(
                    "Refine-bounded candidate -> "
                    f"kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f} "
                    f"(from {kp_raw:.4f}, {ki_raw:.4f}, {kd_raw:.4f})"
                )

        # Keep PID changes smooth by limiting step size from last applied values.
        if trial_index > 0 and last_applied is not None:
            prev_kp, prev_ki, prev_kd = last_applied
            kp_raw, ki_raw, kd_raw = kp, ki, kd
            kp = float(np.clip(kp, prev_kp - step_kp, prev_kp + step_kp))
            ki = float(np.clip(ki, prev_ki - step_ki, prev_ki + step_ki))
            kd = float(np.clip(kd, prev_kd - step_kd, prev_kd + step_kd))
            if (kp, ki, kd) != (kp_raw, ki_raw, kd_raw):
                log(
                    "Step-limited candidate -> "
                    f"kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f} "
                    f"(from {kp_raw:.4f}, {ki_raw:.4f}, {kd_raw:.4f})"
                )

        # If score is flat, switch into local recovery around the best-known point.
        # Recovery first re-tests best PID exactly, then probes small single-axis moves.
        if best_pid is not None and (recovery_mode or no_improve_count >= 3):
            if not recovery_mode:
                recovery_mode = True
                recovery_exact_next = True
                log("Stagnation detected: entering local recovery around best PID")

            bkp, bki, bkd = best_pid
            if recovery_exact_next:
                kp, ki, kd = float(bkp), float(bki), float(bkd)
                recovery_exact_next = False
                log(
                    "Recovery probe -> exact best PID re-test: "
                    f"kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}"
                )
            else:
                kp, ki, kd = float(bkp), float(bki), float(bkd)
                delta_kp = max(min_step_kp, step_kp * 0.35)
                delta_ki = max(min_step_ki, step_ki * 0.35)
                delta_kd = max(min_step_kd, step_kd * 0.35)
                if stagnation_axis == 0:
                    kp = float(np.clip(kp + (stagnation_sign * delta_kp), 0.0, args.kp_max))
                    axis_name = "kp"
                elif stagnation_axis == 1:
                    ki = float(np.clip(ki + (stagnation_sign * delta_ki), 0.0, args.ki_max))
                    axis_name = "ki"
                else:
                    kd = float(np.clip(kd + (stagnation_sign * delta_kd), 0.0, args.kd_max))
                    axis_name = "kd"
                log(
                    "Recovery probe -> best PID with small perturbation "
                    f"on {axis_name}: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}"
                )
                stagnation_axis = (stagnation_axis + 1) % 3
                if stagnation_axis == 0:
                    stagnation_sign *= -1.0

        # Use live laser PID for the first run, then apply optimized candidates.
        apply_pid_update = trial_index > 0
        t, y, u, aborted, current_pid, per_test_powers, per_test_times, per_test_meta = run_trial(
            io,
            kp,
            ki,
            kd,
            desired_output=desired_output,
            apply_pid_update=apply_pid_update,
            test_duration_s=test_duration_s,
            startup_grace_s=args.startup_grace_s,
            settled_window_samples=args.settled_window_samples,
            duration=duration,
            kp_max=args.kp_max,
            ki_max=args.ki_max,
            kd_max=args.kd_max,
        )

        # Record the actual PID values used for this run.
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

        # If we found a new best, shrink step sizes for finer local search.
        # If not, expand slightly (up to configured max) to keep exploring.
        improved = score < prev_best_score
        if improved:
            best_pid = (used_kp, used_ki, used_kd)
            no_improve_count = 0
            recovery_mode = False
            recovery_exact_next = True
            step_kp = max(min_step_kp, step_kp * float(args.step_shrink_factor))
            step_ki = max(min_step_ki, step_ki * float(args.step_shrink_factor))
            step_kd = max(min_step_kd, step_kd * float(args.step_shrink_factor))
        else:
            no_improve_count += 1
            if recovery_mode:
                step_kp = max(min_step_kp, step_kp * 0.95)
                step_ki = max(min_step_ki, step_ki * 0.95)
                step_kd = max(min_step_kd, step_kd * 0.95)
            elif best_improve_pct_so_far < float(args.lock_growth_after_improve_pct):
                step_kp = min(float(args.max_step_kp), step_kp * float(args.step_growth_factor))
                step_ki = min(float(args.max_step_ki), step_ki * float(args.step_growth_factor))
                step_kd = min(float(args.max_step_kd), step_kd * float(args.step_growth_factor))

        if baseline_score and baseline_score > 0:
            improve_vs_base_pct = 100.0 * (baseline_score - score) / baseline_score
            best_improve_vs_base_pct = 100.0 * (baseline_score - best_score_seen) / baseline_score
        else:
            improve_vs_base_pct = 0.0
            best_improve_vs_base_pct = 0.0

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
                int(aborted),
            )
        )

        # Store only the best repeat (out of 5 tests) for this trial.
        # "Best" is lowest mean absolute error to desired output among valid tests.
        best_test_idx = None
        best_test_err = float("inf")
        for i, (test_powers, test_meta) in enumerate(zip(per_test_powers, per_test_meta)):
            if test_powers.size == 0:
                continue
            if bool(test_meta.get("invalid", False)):
                continue
            mae = float(np.mean(np.abs(test_powers - desired_output)))
            if mae < best_test_err:
                best_test_err = mae
                best_test_idx = i

        # Fallback: if all valid tests are unavailable, use the first non-empty one.
        if best_test_idx is None:
            for i, test_powers in enumerate(per_test_powers):
                if test_powers.size > 0:
                    best_test_idx = i
                    break

        if best_test_idx is not None:
            test_idx = best_test_idx + 1
            test_powers = per_test_powers[best_test_idx]
            test_times = per_test_times[best_test_idx]
            test_meta = per_test_meta[best_test_idx]
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

        # Optional best-point re-test for drift/consistency checks.
        if (
            args.retest_best_every > 0
            and trial_index > 0
            and best_pid is not None
            and ((trial_index + 1) % args.retest_best_every == 0)
        ):
            bkp, bki, bkd = best_pid
            log(f"Re-testing best PID at interval -> kp={bkp:.4f}, ki={bki:.4f}, kd={bkd:.4f}")
            _, by, _, baborted, _, btests, btimes, bmeta = run_trial(
                io,
                bkp,
                bki,
                bkd,
                desired_output=desired_output,
                apply_pid_update=True,
                test_duration_s=test_duration_s,
                startup_grace_s=args.startup_grace_s,
                settled_window_samples=args.settled_window_samples,
                duration=duration,
                kp_max=args.kp_max,
                ki_max=args.ki_max,
                kd_max=args.kd_max,
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
            f"aborted={aborted}"
        )

        trial_index += 1
        if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
            raise EarlyStopOptimization(
                f"No score improvement for {no_improve_count} trials (patience={args.early_stop_patience})"
            )
        return score

    # Run Bayesian optimization over kp/ki/kd bounds.
    log("Starting Bayesian Optimisation")
    result = None
    try:
        result = gp_minimize(
            objective,
            space,
            n_calls=n_trials,
            n_initial_points=min(6, n_trials),
            acq_func="EI",
            random_state=42,
        )
    except EarlyStopOptimization as e:
        log(f"Early stop: {e}")

    if best_pid is not None:
        best_kp, best_ki, best_kd = best_pid
    elif result is not None:
        best_kp, best_ki, best_kd = result.x
    else:
        best_kp, best_ki, best_kd = 0.0, 0.0, 0.0
    log("Optimisation complete")
    log(f"BEST kp={best_kp:.6f}, ki={best_ki:.6f}, kd={best_kd:.6f}")
    if best_score_seen < float("inf"):
        log(f"Best score={best_score_seen:.3f}")
    elif result is not None:
        log(f"Best score={result.fun:.3f}")

    # Persist full optimization history for offline analysis.
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


if __name__ == "__main__":
    main()

"""Startup GUI and power trace graphing helpers."""

import csv
from pathlib import Path
import time

import numpy as np


class RuntimeMonitor:
    """Simple live Tk monitor for current power and active PID values."""

    DISPLAY_TIME_OFFSET_S = 0.3

    def __init__(self, root, *, desired_output: float):
        import tkinter as tk

        self.root = root
        self.tk = tk
        self.closed = False
        self.desired_output = float(desired_output)
        self._times: list[float] = []
        self._powers: list[float] = []
        self._last_redraw = 0.0

        for child in self.root.winfo_children():
            child.destroy()

        self.root.title("PID Tuner Monitor")
        self.root.resizable(True, True)
        self.root.geometry("980x620")

        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Preparing tuner...")
        self.phase_var = tk.StringVar(value="Phase: Gathering candidates")
        self.progress_var = tk.StringVar(value="Progress: waiting for first trial")
        self.power_var = tk.StringVar(value="Current power: --")
        self.pid_var = tk.StringVar(value="PID: --")

        tk.Label(frame, text="Live Power Output", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        tk.Label(frame, textvariable=self.phase_var, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(4, 0))
        tk.Label(frame, textvariable=self.progress_var, font=("TkDefaultFont", 10)).pack(anchor="w", pady=(2, 0))
        tk.Label(frame, textvariable=self.status_var).pack(anchor="w", pady=(4, 0))
        tk.Label(frame, textvariable=self.power_var).pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(frame, width=920, height=420, bg="white", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)

        footer = tk.Frame(frame, pady=8)
        footer.pack(fill="x")
        tk.Label(footer, textvariable=self.pid_var, anchor="w", justify="left").pack(side="left")
        self.close_hint_var = tk.StringVar(value="Test window stays open until the run finishes.")
        tk.Label(footer, textvariable=self.close_hint_var, anchor="e", justify="right").pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close_requested)
        self.process_events()
        self._draw_plot()

    def _on_close_requested(self):
        if self.closed:
            return
        self.close_hint_var.set("Use the main menu Quit button to exit the application.")
        self.process_events()

    def process_events(self):
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def set_target(self, desired_output: float):
        self.desired_output = float(desired_output)
        self._draw_plot(force=True)

    def set_status(self, message: str):
        if self.closed:
            return
        self.status_var.set(str(message))
        self.process_events()

    def set_phase(self, phase: str):
        if self.closed:
            return
        self.phase_var.set(str(phase))
        self.process_events()

    def set_progress(self, message: str):
        if self.closed:
            return
        self.progress_var.set(str(message))
        self.process_events()

    def set_pid_values(self, kp: float, ki: float, kd: float):
        if self.closed:
            return
        self.pid_var.set(f"Current PID values: Kp={kp:.4f}  Ki={ki:.4f}  Kd={kd:.4f}")
        self.process_events()

    def begin_test(
        self,
        *,
        phase_name: str,
        phase_trial_index: int,
        phase_trial_total: int | None,
        repeat_index: int,
        repeats: int,
        overall_trial_index: int | None = None,
    ):
        if self.closed:
            return
        self._times = []
        self._powers = []
        if phase_trial_total is not None:
            progress = f"{phase_name} trial {phase_trial_index}/{phase_trial_total} | test {repeat_index}/{repeats}"
        else:
            progress = f"{phase_name} trial {phase_trial_index} | test {repeat_index}/{repeats}"
        if overall_trial_index is not None:
            progress = f"{progress} | overall {overall_trial_index}"
        self.progress_var.set(progress)
        self.power_var.set("Current power: --")
        self._draw_plot(force=True)
        self.process_events()

    def append_sample(self, time_s: float, power: float, *, status: str | None = None):
        if self.closed:
            return
        display_time = float(time_s) + self.DISPLAY_TIME_OFFSET_S
        if not self._times:
            self._times.append(0.0)
            self._powers.append(0.0)
        self._times.append(display_time)
        self._powers.append(float(power))
        self.power_var.set(f"Current power: {float(power):.4f}")
        if status:
            self.status_var.set(str(status))

        now = time.monotonic()
        if (now - self._last_redraw) >= 0.08:
            self._draw_plot()
            self._last_redraw = now
            self.process_events()

    def mark_complete(self, message: str):
        if self.closed:
            return
        self.status_var.set(str(message))
        self._draw_plot(force=True)
        self.process_events()

    def wait_until_closed(self):
        while not self.closed:
            self.process_events()
            time.sleep(0.05)

    def _draw_plot(self, force: bool = False):
        if self.closed:
            return
        canvas = self.canvas
        canvas.delete("plot")

        width = max(int(canvas.winfo_width()), 100)
        height = max(int(canvas.winfo_height()), 100)
        left = 56
        right = width - 18
        top = 18
        bottom = height - 42
        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)

        canvas.create_rectangle(left, top, right, bottom, outline="#808080", tags="plot")
        canvas.create_text(left, bottom + 18, text="0 s", anchor="w", tags="plot")
        canvas.create_text(left - 8, top, text="Power", anchor="e", tags="plot")

        if self._times and self._powers:
            t_min = min(self._times)
            t_max = max(self._times)
            if t_max <= t_min:
                t_max = t_min + 1.0

            y_min = min(min(self._powers), self.desired_output)
            y_max = max(max(self._powers), self.desired_output)
            if y_max <= y_min:
                y_max = y_min + 1.0

            y_pad = max((y_max - y_min) * 0.1, 1.0)
            y_min = min(0.0, y_min - y_pad)
            y_max = y_max + y_pad

            def x_px(value: float) -> float:
                return left + ((value - t_min) / (t_max - t_min)) * plot_w

            def y_px(value: float) -> float:
                return bottom - ((value - y_min) / (y_max - y_min)) * plot_h

            target_y = y_px(self.desired_output)
            canvas.create_line(left, target_y, right, target_y, fill="#cc5533", dash=(5, 3), tags="plot")
            canvas.create_text(right, target_y - 8, text=f"Target {self.desired_output:.2f}", anchor="e", tags="plot")

            points = []
            for t_val, power_val in zip(self._times, self._powers):
                points.extend((x_px(t_val), y_px(power_val)))
            if len(points) >= 4:
                canvas.create_line(*points, fill="#1f77b4", width=2, smooth=True, splinesteps=24, tags="plot")

            canvas.create_text(right, bottom + 18, text=f"{t_max:.2f} s", anchor="e", tags="plot")
            canvas.create_text(left - 8, bottom, text=f"{y_min:.1f}", anchor="e", tags="plot")
            canvas.create_text(left - 8, top, text=f"{y_max:.1f}", anchor="e", tags="plot")
        else:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Waiting for live telemetry...",
                fill="#666666",
                tags="plot",
            )
            if force:
                canvas.create_text(right, bottom + 18, text="0 s", anchor="e", tags="plot")


def prompt_launch_gui(
    default_goal: float,
    default_trials: int,
    default_test_duration_s: float,
    default_frequency_khz: int,
    root=None,
):
    """Show the startup GUI and return selected action + field values."""
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
        "frequency_khz": int(default_frequency_khz),
        "root": root,
    }

    if root is None:
        try:
            root = tk.Tk()
        except Exception:
            return None
    else:
        try:
            root.deiconify()
        except Exception:
            pass

    for child in root.winfo_children():
        child.destroy()

    root.title("PID Tuner")
    root.resizable(False, False)

    frame = tk.Frame(root, padx=12, pady=12)
    frame.grid(row=0, column=0, sticky="nsew")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    tk.Label(frame, text="Goal Power Output").grid(row=0, column=0, sticky="w")
    goal_var = tk.StringVar(value=f"{float(default_goal):.4f}")
    goal_entry = tk.Entry(frame, textvariable=goal_var, width=16)
    goal_entry.grid(row=1, column=0, sticky="w", pady=(2, 8))


    tk.Label(frame, text="Frequency (kHz)").grid(row=2, column=0, sticky="w")
    frequency_var = tk.StringVar(value=str(int(default_frequency_khz)))
    tk.Entry(frame, textvariable=frequency_var, width=16).grid(row=3, column=0, sticky="w", pady=(2, 10))
    
    tk.Label(frame, text="Number of Trials").grid(row=4, column=0, sticky="w")
    trials_var = tk.StringVar(value=str(int(default_trials)))
    tk.Entry(frame, textvariable=trials_var, width=16).grid(row=5, column=0, sticky="w", pady=(2, 10))

    tk.Label(frame, text="Test Duration (s)").grid(row=6, column=0, sticky="w")
    duration_var = tk.StringVar(value=f"{float(default_test_duration_s):.1f}")
    tk.Entry(frame, textvariable=duration_var, width=16).grid(row=7, column=0, sticky="w", pady=(2, 10))

    status_var = tk.StringVar(value="Close the application with the Quit button.")
    tk.Label(frame, textvariable=status_var, fg="#555555").grid(row=9, column=0, sticky="w", pady=(10, 0))

    

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
        try:
            frequency_khz = int(frequency_var.get().strip())
            if frequency_khz < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Frequency must be an integer >= 0 kHz.")
            return False
        result["goal"] = goal
        result["trials"] = trials
        result["test_duration_s"] = test_duration_s
        result["frequency_khz"] = frequency_khz
        return True

    def on_start():
        if not parse_fields():
            return
        result["action"] = "start"
        result["root"] = root
        root.quit()

    def on_reset():
        result["action"] = "reset"
        result["root"] = root
        root.quit()

    def on_quit():
        result["action"] = "quit"
        root.destroy()

    def on_graph():
        result["action"] = "graph"
        result["root"] = root
        root.quit()

    def on_window_close():
        status_var.set("Use the Quit button to close the application.")

    btn_row = tk.Frame(frame)
    btn_row.grid(row=8, column=0, sticky="w")
    tk.Button(btn_row, text="Start Test", width=12, command=on_start).grid(row=0, column=0, padx=(0, 6))
    tk.Button(btn_row, text="Reset Defaults", width=12, command=on_reset).grid(row=0, column=1, padx=(0, 6))
    tk.Button(btn_row, text="Graph Power", width=11, command=on_graph).grid(row=0, column=2, padx=(0, 6))
    tk.Button(btn_row, text="Quit", width=8, command=on_quit).grid(row=0, column=3)

    root.protocol("WM_DELETE_WINDOW", on_window_close)
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
                invalid = bool(int(row.get("test_invalid", "0")))
            except (ValueError, KeyError):
                continue
            key = (trial, test)
            bucket = series_map.setdefault(key, {"time_s": [], "power": [], "goal": goal, "invalid": False})
            bucket["time_s"].append(t_s)
            bucket["power"].append(power)
            bucket["goal"] = goal
            bucket["invalid"] = bucket["invalid"] or invalid

    final = {}
    for key, payload in series_map.items():
        final[key] = {
            "time_s": np.array(payload["time_s"], dtype=float),
            "power": np.array(payload["power"], dtype=float),
            "goal": float(payload["goal"]),
            "invalid": bool(payload["invalid"]),
        }
    return final


def compute_series_mae(payload: dict) -> float:
    """Compute mean absolute error to the goal for one saved test series."""
    power = np.asarray(payload["power"], dtype=float)
    if power.size == 0:
        return float("inf")
    goal = float(payload["goal"])
    return float(np.mean(np.abs(power - goal)))


def find_best_single_test(series: dict):
    """Return the best individual test key by MAE, preferring non-invalid tests."""
    best_key = None
    best_mae = float("inf")
    for key, payload in sorted(series.items()):
        if payload["invalid"]:
            continue
        mae = compute_series_mae(payload)
        if mae < best_mae:
            best_mae = mae
            best_key = key
    if best_key is not None:
        return best_key

    for key, payload in sorted(series.items()):
        mae = compute_series_mae(payload)
        if mae < best_mae:
            best_mae = mae
            best_key = key
    return best_key


def build_trial_average_series(series: dict):
    """Build one averaged trace per trial, dropping the worst valid test as an outlier when possible."""
    tests_by_trial = {}
    for key, payload in series.items():
        tests_by_trial.setdefault(key[0], []).append((key, payload))

    averages = {}
    for trial, entries in sorted(tests_by_trial.items()):
        non_empty = [(key, payload) for key, payload in entries if np.asarray(payload["power"], dtype=float).size > 0]
        if not non_empty:
            continue

        valid = [(key, payload) for key, payload in non_empty if not payload["invalid"]]
        candidates = valid or non_empty
        removed_key = None
        if len(candidates) >= 3:
            removed_key, _ = max(candidates, key=lambda item: compute_series_mae(item[1]))
            candidates = [item for item in candidates if item[0] != removed_key]

        if not candidates:
            candidates = valid or non_empty

        best_test_key, best_test_payload = min(candidates, key=lambda item: compute_series_mae(item[1]))
        min_len = min(
            min(len(np.asarray(payload["time_s"], dtype=float)), len(np.asarray(payload["power"], dtype=float)))
            for _, payload in candidates
        )
        if min_len <= 0:
            continue

        time_stack = np.vstack([np.asarray(payload["time_s"], dtype=float)[:min_len] for _, payload in candidates])
        power_stack = np.vstack([np.asarray(payload["power"], dtype=float)[:min_len] for _, payload in candidates])
        averages[trial] = {
            "time_s": np.mean(time_stack, axis=0),
            "power": np.mean(power_stack, axis=0),
            "goal": float(best_test_payload["goal"]),
            "source_tests": [key[1] for key, _ in candidates],
            "removed_test": None if removed_key is None else removed_key[1],
            "best_test_key": best_test_key,
        }

    return averages


def load_best_trial_index(history_csv_path: str):
    """Return the 1-based trial index with the lowest score, if available."""
    best_trial = None
    best_score = None
    with open(history_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            try:
                score = float(row["score"])
            except (ValueError, KeyError):
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_trial = row_index
    return best_trial


def infer_history_csv_path(power_csv_path: str) -> str:
    """Infer the sibling tuning history CSV path from the power readings CSV path."""
    power_path = Path(power_csv_path)
    if power_path.name == "tuning_power_readings.csv":
        return str(power_path.with_name("tuning_history.csv"))
    return str(power_path.with_name(f"{power_path.stem}_history.csv"))


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
    """Open graph window showing only the initial and best trial traces."""
    import matplotlib.pyplot as plt

    series = load_power_series(csv_path)
    if not series:
        raise RuntimeError(f"No valid readings found in {csv_path}")

    best_trial_index = None
    history_csv_path = infer_history_csv_path(csv_path)
    try:
        best_trial_index = load_best_trial_index(history_csv_path)
    except FileNotFoundError:
        best_trial_index = None
    trial_averages = build_trial_average_series(series)
    if not trial_averages:
        raise RuntimeError(f"No valid trial averages found in {csv_path}")

    initial_trial_index = min(trial_averages.keys())
    selected_trials = [initial_trial_index]
    if best_trial_index is not None and best_trial_index in trial_averages and best_trial_index != initial_trial_index:
        selected_trials.append(best_trial_index)

    fig, ax_plot = plt.subplots(figsize=(12, 6))

    goals_plotted = set()
    for trial in selected_trials:
        payload = trial_averages[trial]
        removed_test = payload["removed_test"]
        if trial == initial_trial_index:
            label = "Initial trial"
        else:
            label = "Best trial"
        if removed_test is not None:
            label = f"{label} (excluding Test {removed_test})"
        if trial == best_trial_index and trial == initial_trial_index:
            label = "Initial / Best trial"
        ax_plot.plot(
            payload["time_s"],
            payload["power"],
            label=label,
            linewidth=2.8,
        )
        goal_val = float(payload["goal"])
        if goal_val not in goals_plotted:
            ax_plot.axhline(goal_val, linestyle="--", linewidth=1.0, alpha=0.5, label=f"Goal {goal_val:.2f}")
            goals_plotted.add(goal_val)

    def apply_default_view():
        ax_plot.relim()
        ax_plot.autoscale_view(scalex=True, scaley=True)
        y_top = ax_plot.get_ylim()[1]
        if not np.isfinite(y_top) or y_top <= 0.0:
            y_top = 1.0
        ax_plot.set_ylim(bottom=0.0, top=y_top)

    title = "Power Readings: Initial vs Best Trial"
    if len(selected_trials) == 1:
        title = "Power Readings: Initial Trial"
    ax_plot.set_title(title)
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

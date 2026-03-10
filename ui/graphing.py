"""Startup GUI and power trace graphing helpers."""

import csv
import time

import numpy as np


class RuntimeMonitor:
    """Simple live Tk monitor for current power and active PID values."""

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
        self.power_var = tk.StringVar(value="Current power: --")
        self.pid_var = tk.StringVar(value="PID: --")

        tk.Label(frame, text="Live Power Output", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        tk.Label(frame, textvariable=self.status_var).pack(anchor="w", pady=(4, 0))
        tk.Label(frame, textvariable=self.power_var).pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(frame, width=920, height=420, bg="white", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)

        footer = tk.Frame(frame, pady=8)
        footer.pack(fill="x")
        tk.Label(footer, textvariable=self.pid_var, anchor="w", justify="left").pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.process_events()
        self._draw_plot()

    def _on_close(self):
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

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

    def set_pid_values(self, kp: float, ki: float, kd: float):
        if self.closed:
            return
        self.pid_var.set(f"Current PID values: Kp={kp:.4f}  Ki={ki:.4f}  Kd={kd:.4f}")
        self.process_events()

    def begin_test(self, *, trial_index: int, repeat_index: int, repeats: int):
        if self.closed:
            return
        self._times = []
        self._powers = []
        self.status_var.set(f"Trial {trial_index}: test {repeat_index}/{repeats} running")
        self.power_var.set("Current power: --")
        self._draw_plot(force=True)
        self.process_events()

    def append_sample(self, time_s: float, power: float, *, status: str | None = None):
        if self.closed:
            return
        self._times.append(float(time_s))
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
                canvas.create_line(*points, fill="#1f77b4", width=2, tags="plot")

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
        "root": None,
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


    tk.Label(frame, text="Frequency (kHz)").grid(row=2, column=0, sticky="w")
    frequency_var = tk.StringVar(value=str(int(default_frequency_khz)))
    tk.Entry(frame, textvariable=frequency_var, width=16).grid(row=3, column=0, sticky="w", pady=(2, 10))
    
    tk.Label(frame, text="Number of Trials").grid(row=4, column=0, sticky="w")
    trials_var = tk.StringVar(value=str(int(default_trials)))
    tk.Entry(frame, textvariable=trials_var, width=16).grid(row=5, column=0, sticky="w", pady=(2, 10))

    tk.Label(frame, text="Test Duration (s)").grid(row=6, column=0, sticky="w")
    duration_var = tk.StringVar(value=f"{float(default_test_duration_s):.1f}")
    tk.Entry(frame, textvariable=duration_var, width=16).grid(row=7, column=0, sticky="w", pady=(2, 10))

    

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
        root.destroy()

    def on_quit():
        result["action"] = "quit"
        root.destroy()

    def on_graph():
        result["action"] = "graph"
        root.destroy()

    btn_row = tk.Frame(frame)
    btn_row.grid(row=8, column=0, sticky="w")
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
        ax_plot.relim()
        ax_plot.autoscale_view(scalex=True, scaley=True)
        y_top = ax_plot.get_ylim()[1]
        if not np.isfinite(y_top) or y_top <= 0.0:
            y_top = 1.0
        ax_plot.set_ylim(bottom=0.0, top=y_top)

    def apply_zoomed_view():
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

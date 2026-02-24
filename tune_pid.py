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


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_trial(io: SerialLineIO, kp: float, ki: float, kd: float, setpoint: float = 0.8, duration: float = 8.0):
    """
    Runs one tuning trial and returns arrays (t, y, u) plus aborted flag.
    """
    _ = duration
    log(f"Starting trial: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

    kp = float(np.clip(kp, 0.0, 10.0))
    ki = float(np.clip(ki, 0.0, 10.0))
    kd = float(np.clip(kd, 0.0, 2.0))

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

    io.write_command("", command_id_hex2=CMD.START)

    t_vals, y_vals, u_vals, status_vals = collect_trial_data(
        io,
        line_timeout=5.0,
        on_done=lambda: log("Trial finished"),
    )

    aborted = any(s == "ABORT" for s in status_vals)
    if aborted:
        log("Warning: Trial aborted due to safety condition")

    return np.array(t_vals), np.array(y_vals), np.array(u_vals), aborted


def compute_metrics(t: np.ndarray, y: np.ndarray, setpoint: float, settle_band: float = 0.02):
    if len(t) < 5:
        return 999.0, 999.0, 999.0

    error = setpoint - y
    iae = float(np.trapezoid(np.abs(error), t))

    peak = float(np.max(y))
    overshoot = 0.0
    if peak > setpoint and setpoint != 0:
        overshoot = 100.0 * (peak - setpoint) / abs(setpoint)

    band_low = setpoint * (1.0 - settle_band)
    band_high = setpoint * (1.0 + settle_band)
    settling_time = float(t[-1])
    for i in range(len(t)):
        if np.all((y[i:] >= band_low) & (y[i:] <= band_high)):
            settling_time = float(t[i])
            break

    return overshoot, settling_time, iae


def score_controller(overshoot: float, settling_time: float, iae: float, aborted: bool):
    score = 0.0
    score += 2.0 * overshoot
    score += 1.5 * settling_time
    score += 1.0 * iae
    if aborted:
        score += 500.0
    return float(score)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Serial port (e.g. /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--iters", type=int, default=20, help="Number of optimisation iterations")
    ap.add_argument("--log-data", action="store_true", help="Log some DATA lines too (can be spammy)")
    ap.add_argument("--log-data-every", type=int, default=50, help="If --log-data, log every Nth DATA line")
    args = ap.parse_args()

    log("Opening serial port")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    io = SerialLineIO(
        ser,
        log_fn=log,
        log_data_lines=args.log_data,
        data_log_every=args.log_data_every,
    )

    io.write_command("123", command_id_hex2=CMD.PING)
    resp = io.read_line(timeout=2.0)
    if "PONG" not in resp:
        log("Warning: unexpected PING response, continuing anyway")

    setpoint = 0.8
    duration = 15.0
    space = [
        Real(0.0, 6.0, name="kp"),
        Real(0.0, 6.0, name="ki"),
        Real(0.0, 1.0, name="kd"),
    ]

    history = []

    def objective(x):
        kp, ki, kd = x
        t, y, u, aborted = run_trial(io, kp, ki, kd, setpoint=setpoint, duration=duration)
        overshoot, settling, iae = compute_metrics(t, y, setpoint)
        score = score_controller(overshoot, settling, iae, aborted)
        history.append((kp, ki, kd, score, overshoot, settling, iae, int(aborted)))
        log(
            f"Result -> score={score:.2f}, "
            f"overshoot={overshoot:.2f}%, "
            f"settling={settling:.3f}s, "
            f"IAE={iae:.3f}, "
            f"aborted={aborted}"
        )
        _ = u
        return score

    log("Starting Bayesian Optimisation")
    result = gp_minimize(
        objective,
        space,
        n_calls=args.iters,
        n_initial_points=min(6, args.iters),
        acq_func="EI",
        random_state=42,
    )

    best_kp, best_ki, best_kd = result.x
    log("Optimisation complete")
    log(f"BEST kp={best_kp:.6f}, ki={best_ki:.6f}, kd={best_kd:.6f}")
    log(f"Best score={result.fun:.3f}")

    with open("tuning_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kp", "ki", "kd", "score", "overshoot_pct", "settling_s", "iae", "aborted"])
        writer.writerows(history)
    log("Saved tuning_history.csv")


if __name__ == "__main__":
    main()

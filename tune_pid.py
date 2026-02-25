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


def prompt_launch_action() -> str:
    """
    Ask user whether to start tuning, test protocol command, or exit.
    Returns "start", "test-protocol", or "quit".
    """
    while True:
        choice = input("Choose action: [s]tart test, [t]est protocol, or [q]uit: ").strip().lower()
        if choice in {"s", "start"}:
            return "start"
        if choice in {"t", "test", "protocol", "test protocol"}:
            return "test-protocol"
        if choice in {"q", "quit", "exit"}:
            return "quit"
        print("Please enter 's' to start, 't' to test protocol, or 'q' to quit.", flush=True)


def run_trial(
    io: SerialLineIO,
    kp: float,
    ki: float,
    kd: float,
    duration: float = 8.0,
    kp_max: float = 1.0,
    ki_max: float = 1.0,
    kd_max: float = 0.2,
):
    """
    Runs one tuning trial and returns arrays (t, y, u) plus aborted flag.
    """
    _ = duration
    log(f"Starting trial: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

    kp = float(np.clip(kp, 0.0, kp_max))
    ki = float(np.clip(ki, 0.0, ki_max))
    kd = float(np.clip(kd, 0.0, kd_max))

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

    sample_interval_s = None
    if current_pid is not None:
        raw_si = float(current_pid.get("sample_interval", 0.0))
        if raw_si > 0:
            # Common firmware behavior: values > 1 are in milliseconds.
            sample_interval_s = raw_si / 1000.0 if raw_si > 1.0 else raw_si
            log(f"Using telemetry sample interval: {sample_interval_s:.6f}s")

    # ack = io.set_pid_values(
    #     pw_kp=kp,
    #     pw_ki=ki,
    #     pw_kd=kd,
    #     current_values=current_pid,
    #     timeout=2.0,
    # )
    # ok_ack, _ = parse_ack(ack)
    # if not ack.startswith("*"):
    #     log(f"Warning: Unexpected SET_PID acknowledgment: {ack}")
    # elif not ok_ack:
    #     log(f"Warning: SET_PID returned error code: {ack}")

    io.write_command("", command_id_hex2=CMD.RUN)
    io.write_command("1", command_id_hex2=CMD.SHUTTER_CONTROL)
    io.write_command("", command_id_hex2=CMD.TRIGGER)

    t_vals, y_vals, u_vals, status_vals = collect_trial_data(
        io,
        line_timeout=5.0,
        sample_interval_s=sample_interval_s,
        on_done=lambda: log("Trial finished"),
    )

    aborted = any(s == "ABORT" for s in status_vals)
    if aborted:
        log("Warning: Trial aborted due to safety condition")

    return np.array(t_vals), np.array(y_vals), np.array(u_vals), aborted


def compute_metrics(y: np.ndarray, desired_output: float):
    if len(y) < 5:
        return 999.0, 999.0, 999.0

    error = y - desired_output
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    error_std = float(np.std(error))
    return mae, rmse, error_std


def score_controller(mae: float, error_std: float, aborted: bool):
    score = mae + error_std
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
    ap.add_argument("--kp-max", type=float, default=1.0, help="Upper limit for Kp search/clamp")
    ap.add_argument("--ki-max", type=float, default=1.0, help="Upper limit for Ki search/clamp")
    ap.add_argument("--kd-max", type=float, default=0.2, help="Upper limit for Kd search/clamp")
    ap.add_argument("--desired-output", type=float, default=0.8, help="Target output value for scoring")
    args = ap.parse_args()

    action = prompt_launch_action()
    if action == "quit":
        log("Exiting on user request.")
        return

    log("Opening serial port")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    io = SerialLineIO(
        ser,
        log_fn=log,
        log_data_lines=args.log_data,
        data_log_every=args.log_data_every,
    )

    if action == "test-protocol":
        log("Sending GET_FLOW test command")
        io.write_command("", command_id_hex2=CMD.GET_FLOW)
        try:
            resp = io.read_line(timeout=2.0)
            log(f"GET_FLOW response: {resp}")
        except Exception as e:
            log(f"No immediate GET_FLOW response: {e}")
        log("Protocol test complete. Exiting.")
        return

    duration = 15.0
    space = [
        Real(0.0, args.kp_max, name="kp"),
        Real(0.0, args.ki_max, name="ki"),
        Real(0.0, args.kd_max, name="kd"),
    ]

    history = []

    def objective(x):
        kp, ki, kd = x
        t, y, u, aborted = run_trial(
            io,
            kp,
            ki,
            kd,
            duration=duration,
            kp_max=args.kp_max,
            ki_max=args.ki_max,
            kd_max=args.kd_max,
        )
        mae, rmse, error_std = compute_metrics(y, args.desired_output)
        score = score_controller(mae, error_std, aborted)
        history.append((kp, ki, kd, score, mae, rmse, error_std, int(aborted)))
        log(
            f"Result -> score={score:.2f}, "
            f"MAE={mae:.5f}, "
            f"RMSE={rmse:.5f}, "
            f"err_std={error_std:.5f}, "
            f"aborted={aborted}"
        )
        _ = u
        _ = t
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
        writer.writerow(["kp", "ki", "kd", "score", "mae", "rmse", "error_std", "aborted"])
        writer.writerows(history)
    log("Saved tuning_history.csv")


if __name__ == "__main__":
    main()

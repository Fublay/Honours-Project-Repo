import time
import re
import csv
import serial
import numpy as np
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real

# ---------------------------------------------------------------------------
# Simple console logger
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------------------------------------------------------
# Regex used to parse telemetry lines sent by the laser simulator.
# Example:
#   DATA t=0.1200 y=0.734000 sp=0.800000 u=0.512000 status=OK
# ---------------------------------------------------------------------------
DATA_RE = re.compile(
    r"t=([0-9.]+)\s+"
    r"y=([0-9.eE+-]+)\s+"
    r"sp=([0-9.eE+-]+)\s+"
    r"u=([0-9.eE+-]+)\s+"
    r"status=([A-Z]+)"
)

# ---------------------------------------------------------------------------
# Robust serial line reader/writer
# Keeps leftover bytes between calls so we never lose data.
# This fixes the exact issue you hit (garbled/partial RX lines and timeouts).
# ---------------------------------------------------------------------------
class SerialLineIO:
    def __init__(self, ser: serial.Serial, log_data_lines: bool = False, data_log_every: int = 50):
        """
        log_data_lines:
            If True, will log some DATA lines too (can be very spammy).
        data_log_every:
            If log_data_lines True, only logs every Nth DATA line.
        """
        self.ser = ser
        self.buf = b""
        self.log_data_lines = log_data_lines
        self.data_log_every = max(1, int(data_log_every))
        self._data_seen = 0

    def write_line(self, line: str) -> None:
        log(f"TX → {line}")
        self.ser.write((line.strip() + "\n").encode("ascii", errors="ignore"))

    def read_line(self, timeout: float = 2.0) -> str:
        t0 = time.time()

        while time.time() - t0 < timeout:
            # If we already have a full line buffered, return it.
            if b"\n" in self.buf:
                line_bytes, self.buf = self.buf.split(b"\n", 1)
                line = line_bytes.decode("ascii", errors="ignore").strip()

                # Logging policy: always log OK/ERR; optionally log some DATA
                if line.startswith("DATA"):
                    self._data_seen += 1
                    if self.log_data_lines and (self._data_seen % self.data_log_every == 0):
                        log(f"RX ← {line}")
                else:
                    log(f"RX ← {line}")

                return line

            # Otherwise, read more bytes from serial.
            chunk = self.ser.read(256)
            if chunk:
                self.buf += chunk
            else:
                time.sleep(0.01)

        raise TimeoutError("Timed out waiting for serial data")

# ---------------------------------------------------------------------------
# Run ONE closed-loop experiment
# ---------------------------------------------------------------------------
def run_trial(io: SerialLineIO, kp: float, ki: float, kd: float, setpoint: float = 0.8, duration: float = 8.0):
    """
    Runs one tuning trial and returns arrays (t, y, u) plus aborted flag.
    """

    log(f"Starting trial: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

    # HARD SAFETY LIMITS for gains (controller-side clamp)
    kp = float(np.clip(kp, 0.0, 10.0))
    ki = float(np.clip(ki, 0.0, 10.0))
    kd = float(np.clip(kd, 0.0, 2.0))

    # Apply PID
    io.write_line(f"SET PID {kp} {ki} {kd}")
    io.read_line(timeout=2.0)

    # Apply setpoint
    io.write_line(f"SET SP {setpoint}")
    io.read_line(timeout=2.0)

    # Start trial
    io.write_line(f"START {duration}")

    t_vals, y_vals, u_vals, status_vals = [], [], [], []

    while True:
        line = io.read_line(timeout=5.0)

        if line.startswith("DATA"):
            match = DATA_RE.search(line)
            if not match:
                # Ignore malformed telemetry lines
                continue

            t, y, sp, u, status = match.groups()
            t_vals.append(float(t))
            y_vals.append(float(y))
            u_vals.append(float(u))
            status_vals.append(status)

        elif line.startswith("OK DONE"):
            log("Trial finished")
            break

        elif line.startswith("ERR"):
            raise RuntimeError(line)

        # Any other lines are ignored (or already logged above)

    aborted = any(s == "ABORT" for s in status_vals)
    if aborted:
        log("⚠ Trial aborted due to safety condition")

    return np.array(t_vals), np.array(y_vals), np.array(u_vals), aborted

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(t: np.ndarray, y: np.ndarray, setpoint: float, settle_band: float = 0.02):
    """
    Computes:
      - overshoot (%)
      - settling time (s)
      - integral of absolute error (IAE)
    """
    if len(t) < 5:
        return 999.0, 999.0, 999.0

    error = setpoint - y

    # Integral of Absolute Error
    iae = float(np.trapezoid(np.abs(error), t))

    # Overshoot
    peak = float(np.max(y))
    overshoot = 0.0
    if peak > setpoint and setpoint != 0:
        overshoot = 100.0 * (peak - setpoint) / abs(setpoint)

    # Settling time: first time after which y stays within band
    band_low = setpoint * (1.0 - settle_band)
    band_high = setpoint * (1.0 + settle_band)

    settling_time = float(t[-1])
    for i in range(len(t)):
        if np.all((y[i:] >= band_low) & (y[i:] <= band_high)):
            settling_time = float(t[i])
            break

    return overshoot, settling_time, iae

# ---------------------------------------------------------------------------
# Objective scoring
# ---------------------------------------------------------------------------
def score_controller(overshoot: float, settling_time: float, iae: float, aborted: bool):
    """
    Combines metrics into one scalar score. Lower is better.
    """
    score = 0.0
    score += 2.0 * overshoot
    score += 1.5 * settling_time
    score += 1.0 * iae

    if aborted:
        score += 500.0

    return float(score)

# ---------------------------------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------------------------------
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

    # Create robust line IO wrapper
    io = SerialLineIO(ser, log_data_lines=args.log_data, data_log_every=args.log_data_every)

    # Connectivity check
    io.write_line("PING")
    resp = io.read_line(timeout=2.0)
    if "PONG" not in resp:
        log("⚠ Warning: unexpected PING response, continuing anyway")

    setpoint = 0.8
    duration = 15.0

    # Search space for PID gains
    space = [
        Real(0.0, 6.0, name="kp"),
        Real(0.0, 6.0, name="ki"),
        Real(0.0, 1.0, name="kd"),
    ]

    history = []  # (kp, ki, kd, score, overshoot, settling, iae, aborted)

    def objective(x):
        kp, ki, kd = x

        t, y, u, aborted = run_trial(io, kp, ki, kd, setpoint=setpoint, duration=duration)

        overshoot, settling, iae = compute_metrics(t, y, setpoint)
        score = score_controller(overshoot, settling, iae, aborted)

        history.append((kp, ki, kd, score, overshoot, settling, iae, int(aborted)))

        log(
            f"Result → score={score:.2f}, "
            f"overshoot={overshoot:.2f}%, "
            f"settling={settling:.3f}s, "
            f"IAE={iae:.3f}, "
            f"aborted={aborted}"
        )

        return score

    log("Starting Bayesian Optimisation")
    result = gp_minimize(
        objective,
        space,
        n_calls=args.iters,
        n_initial_points=min(6, args.iters),
        acq_func="EI",
        random_state=42
    )

    best_kp, best_ki, best_kd = result.x

    log("Optimisation complete")
    log(f"BEST kp={best_kp:.6f}, ki={best_ki:.6f}, kd={best_kd:.6f}")
    log(f"Best score={result.fun:.3f}")

    # Save results
    with open("tuning_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kp", "ki", "kd", "score", "overshoot_pct", "settling_s", "iae", "aborted"])
        writer.writerows(history)

    log("Saved tuning_history.csv")

if __name__ == "__main__":
    main()
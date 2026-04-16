"""Read live telemetry lines and turn them into trial-ready arrays.

This file is intentionally small: it owns the read loop and leaves protocol
parsing/mapping to the dedicated protocol/domain layers.
"""

# Keep the collection loop isolated from trial orchestration so serial timing
# concerns stay in one place.

import time

from protocol.reply_parser import parse_telemetry_line
from domain.value_mapper import map_telemetry_values


# Distinguish "no telemetry ever arrived after START" from a more generic
# empty-trace failure later in the scoring path.
class StartupTelemetryTimeoutError(TimeoutError):
    """Raised when no valid telemetry sample arrives after START within the allowed window."""


# This loop only knows how to read, parse, and timestamp samples.
# Higher-level trial code decides what counts as an invalid or early-stopped run
# through the callbacks.
def collect_trial_data(
    io,
    *,
    line_timeout: float = 1.0,
    sample_interval_s: float | None = None,
    duration_s: float | None = None,
    wait_for_first_sample_timeout_s: float | None = None,
    stop_on_done: bool = True,
    on_sample=None,
    on_done=None,
    on_waiting_for_first_sample=None,
    on_first_sample=None,
):
    """Collect one block of telemetry samples from the serial stream.

    Output format:
    - t_vals: time values per sample
    - y_vals: measured process values (power)
    - u_vals: control output values
    - status_vals: status text per sample

    Stop conditions:
    - fixed duration has elapsed (if duration_s is set), or
    - device says "OK DONE" and stop_on_done=True, or
    - callback requests early stop.
    """
    # We build plain Python lists first because append-in-a-loop is fast/safe.
    t_vals, y_vals, u_vals, status_vals = [], [], [], []
    sample_idx = 0
    # Do not start the scored window until a real telemetry sample has been
    # parsed successfully.
    t_start = None
    wait_started_at = time.monotonic()
    first_sample_time_s = None

    if on_waiting_for_first_sample is not None:
        on_waiting_for_first_sample()

    while True:
        # Time-window mode: leave once the requested capture time has passed.
        now = time.monotonic()
        if t_start is not None and duration_s is not None and (now - t_start) >= duration_s:
            break
        if (
            t_start is None
            and wait_for_first_sample_timeout_s is not None
            and (now - wait_started_at) >= wait_for_first_sample_timeout_s
        ):
            raise StartupTelemetryTimeoutError("no telemetry received after START")

        try:
            line = io.read_line(timeout=line_timeout)
        except TimeoutError:
            # Timeouts are expected sometimes on serial links, so keep waiting.
            continue

        line_s = line.strip()

        # Try to decode this line as a telemetry sample.
        telemetry = parse_telemetry_line(line_s)
        if telemetry is not None:
            mapped = map_telemetry_values(telemetry)
            mapped_t = mapped.get("time_s")
            if t_start is None:
                # First parsed sample flips the collector from "arming" into
                # "timed capture" mode.
                t_start = time.monotonic()
                if mapped_t is not None:
                    first_sample_time_s = float(mapped_t)
                if on_first_sample is not None:
                    on_first_sample(float(t_start - wait_started_at), mapped)

            if mapped_t is None:
                # Some packet formats do not include explicit time.
                # In that case, synthesize time from sample interval if known.
                if sample_interval_s is not None and sample_interval_s > 0:
                    t_val = float(sample_idx) * float(sample_interval_s)
                else:
                    # Last fallback: use sample count as a coarse timeline.
                    t_val = float(sample_idx)
            else:
                if first_sample_time_s is None:
                    first_sample_time_s = float(mapped_t)
                # Rebase device timestamps so the measured window starts at 0.
                t_val = float(mapped_t) - float(first_sample_time_s)

            t_vals.append(t_val)
            y_vals.append(mapped["process_value"])
            u_vals.append(mapped["control_output"])
            status_vals.append(mapped["status"])
            sample_idx += 1

            if on_sample is not None:
                # Caller can stop collection immediately (for safety conditions).
                stop_now = bool(on_sample(t_val, mapped))
                if stop_now:
                    break
            continue

        # Some firmware flows indicate trial completion with this line.
        if line_s.startswith("OK DONE"):
            if on_done:
                on_done()
            if stop_on_done:
                break
            continue

        # Device-reported errors should stop the trial immediately.
        if line_s.startswith("ERR"):
            raise RuntimeError(line_s)

    # Leave conversion to numpy arrays to the caller so this helper stays
    # lightweight and easy to test.
    return t_vals, y_vals, u_vals, status_vals

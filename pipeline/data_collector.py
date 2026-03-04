"""Read live telemetry lines and turn them into trial-ready arrays.

This file is intentionally small: it owns the read loop and leaves protocol
parsing/mapping to the dedicated protocol/domain layers.
"""

import time

from protocol.reply_parser import parse_telemetry_line
from domain.value_mapper import map_telemetry_values


def collect_trial_data(
    io,
    *,
    line_timeout: float = 1.0,
    sample_interval_s: float | None = None,
    duration_s: float | None = None,
    stop_on_done: bool = True,
    on_sample=None,
    on_done=None,
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
    t_start = time.monotonic()

    while True:
        # Time-window mode: leave once the requested capture time has passed.
        if duration_s is not None and (time.monotonic() - t_start) >= duration_s:
            break

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
            if mapped_t is None:
                # Some packet formats do not include explicit time.
                # In that case, synthesize time from sample interval if known.
                if sample_interval_s is not None and sample_interval_s > 0:
                    t_val = float(sample_idx) * float(sample_interval_s)
                else:
                    # Last fallback: use sample count as a coarse timeline.
                    t_val = float(sample_idx)
            else:
                t_val = float(mapped_t)

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

    return t_vals, y_vals, u_vals, status_vals

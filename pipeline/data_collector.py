from protocol.reply_parser import parse_telemetry_line
from domain.value_mapper import map_telemetry_values


def collect_trial_data(io, *, line_timeout: float = 5.0, sample_interval_s: float | None = None, on_done=None):
    """
    Collect trial telemetry until 'OK DONE' or raise on 'ERR...'.
    Returns (t_vals, y_vals, u_vals, status_vals).

    If telemetry does not include explicit time, time is synthesized using
    sample_interval_s (or sample index fallback).
    """
    t_vals, y_vals, u_vals, status_vals = [], [], [], []
    sample_idx = 0

    while True:
        line = io.read_line(timeout=line_timeout)
        line_s = line.strip()

        telemetry = parse_telemetry_line(line_s)
        if telemetry is not None:
            mapped = map_telemetry_values(telemetry)
            mapped_t = mapped.get("time_s")
            if mapped_t is None:
                if sample_interval_s is not None and sample_interval_s > 0:
                    t_val = float(sample_idx) * float(sample_interval_s)
                else:
                    t_val = float(sample_idx)
            else:
                t_val = float(mapped_t)

            t_vals.append(t_val)
            y_vals.append(mapped["process_value"])
            u_vals.append(mapped["control_output"])
            status_vals.append(mapped["status"])
            sample_idx += 1
            continue

        if line_s.startswith("OK DONE"):
            if on_done:
                on_done()
            break

        if line_s.startswith("ERR"):
            raise RuntimeError(line_s)

    return t_vals, y_vals, u_vals, status_vals

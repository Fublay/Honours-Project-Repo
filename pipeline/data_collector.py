from protocol.reply_parser import parse_telemetry_line
from domain.value_mapper import map_telemetry_values


def collect_trial_data(io, *, line_timeout: float = 5.0, on_done=None):
    """
    Collect trial telemetry until 'OK DONE' or raise on 'ERR...'.
    Returns (t_vals, y_vals, u_vals, status_vals).
    """
    t_vals, y_vals, u_vals, status_vals = [], [], [], []

    while True:
        line = io.read_line(timeout=line_timeout)
        line_s = line.strip()

        telemetry = parse_telemetry_line(line_s)
        if telemetry is not None:
            mapped = map_telemetry_values(telemetry)
            t_vals.append(mapped["time_s"])
            y_vals.append(mapped["process_value"])
            u_vals.append(mapped["control_output"])
            status_vals.append(mapped["status"])
            continue

        if line_s.startswith("OK DONE"):
            if on_done:
                on_done()
            break

        if line_s.startswith("ERR"):
            raise RuntimeError(line_s)

    return t_vals, y_vals, u_vals, status_vals

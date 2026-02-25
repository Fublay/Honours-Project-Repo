from protocol.frame_codec import compose_frame, default_checksum_hex_2


def format_pid_value(value: float, field_width: int = 8) -> str:
    """
    Format a PID parameter value to a fixed-width string.
    """
    if abs(value) < 0.001:
        formatted = "0.0000"
    elif abs(value) < 10:
        formatted = f"{value:.4f}"
    elif abs(value) < 100:
        formatted = f"{value:.2f}"
    else:
        formatted = f"{value:.2f}"

    return f"{formatted:>{field_width}}"


def compose_set_pid_command(
    pw_kp: float,
    pw_ki: float,
    pw_kd: float,
    pp_kp: float | None = None,
    pp_ki: float | None = None,
    pp_kd: float | None = None,
    holdoff: float | None = None,
    sample_interval: float | None = None,
    current_values: dict | None = None,
    checksum_fn=default_checksum_hex_2,
) -> bytes:
    """
    Compose a SET_PID command frame.
    """
    if current_values:
        pp_kp = pp_kp if pp_kp is not None else current_values.get("pp_kp", 0.15)
        pp_ki = pp_ki if pp_ki is not None else current_values.get("pp_ki", 0.14)
        pp_kd = pp_kd if pp_kd is not None else current_values.get("pp_kd", 0.05)
        holdoff = holdoff if holdoff is not None else current_values.get("holdoff", 400.0)
        sample_interval = sample_interval if sample_interval is not None else current_values.get("sample_interval", 300.0)
    else:
        pp_kp = pp_kp if pp_kp is not None else 0.15
        pp_ki = pp_ki if pp_ki is not None else 0.14
        pp_kd = pp_kd if pp_kd is not None else 0.05
        holdoff = holdoff if holdoff is not None else 400.0
        sample_interval = sample_interval if sample_interval is not None else 300.0

    data_parts = [
        format_pid_value(pw_kp),
        format_pid_value(pw_ki),
        format_pid_value(pw_kd),
        format_pid_value(pp_kp),
        format_pid_value(pp_ki),
        format_pid_value(pp_kd),
        format_pid_value(holdoff),
        format_pid_value(sample_interval),
    ]
    # Fixed-width protocol payload: 8 chars per field, left padded with spaces.
    data = "".join(data_parts)
    return compose_frame("B5", data, checksum_fn=checksum_fn)

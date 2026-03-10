"""Build protocol payloads for controller commands.

This module focuses on formatting rules the firmware expects, especially the
fixed-width PID fields used by SET_PID.
"""

from protocol.frame_codec import compose_frame, default_checksum_hex_2


def format_pid_value(value: float, field_width: int = 8) -> str:
    """Convert a number to the exact text layout expected by the laser.

    The firmware parser expects each field to be right-aligned in a fixed
    width, so spacing is part of the protocol and must be preserved.
    """
    # Very tiny values are rounded to explicit zero for stable formatting.
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
    """Create one complete SET_PID command frame ready to send on serial."""
    # If caller provided current values, use them as defaults for fields the
    # caller did not explicitly override. This avoids accidental zeroing.
    if current_values:
        pp_kp = pp_kp if pp_kp is not None else current_values.get("pp_kp", 0.15)
        pp_ki = pp_ki if pp_ki is not None else current_values.get("pp_ki", 0.14)
        pp_kd = pp_kd if pp_kd is not None else current_values.get("pp_kd", 0.05)
        holdoff = holdoff if holdoff is not None else current_values.get("holdoff", 400.0)
        sample_interval = sample_interval if sample_interval is not None else current_values.get("sample_interval", 300.0)
    else:
        # Hard defaults used when we have no current PID reply to merge with.
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


def compose_set_program_command(
    *,
    power_w: float | None = None,
    frequency_khz: float | None = None,
    program_id: int | None = 0,
    pulse_width_us: int | None = 0,
    detect_delay_us: int | None = 0,
    current_values: dict | None = None,
    checksum_fn=default_checksum_hex_2,
) -> bytes:
    """Create one complete SET_PROGRAM command frame ready to send on serial."""
    if current_values:
        program_id = current_values.get("program_id", 0) if program_id is None else program_id
        power_w = current_values.get("power_w", 0) if power_w is None else power_w
        frequency_khz = current_values.get("frequency_khz", 0) if frequency_khz is None else frequency_khz
        pulse_width_us = current_values.get("pulse_width_us", 0) if pulse_width_us is None else pulse_width_us
        detect_delay_us = current_values.get("detect_delay_us", 0) if detect_delay_us is None else detect_delay_us

    if power_w is None or frequency_khz is None:
        raise ValueError("power_w and frequency_khz are required when current_values does not provide them")

    program_id_int = int(program_id if program_id is not None else 0)
    power_int = int(round(float(power_w)))
    frequency_int = int(round(float(frequency_khz)))
    pulse_width_int = int(pulse_width_us if pulse_width_us is not None else 0)
    detect_delay_int = int(detect_delay_us if detect_delay_us is not None else 0)

    if not (0 <= program_id_int <= 99):
        raise ValueError("program_id must fit in AA (00-99)")
    if not (0 <= power_int <= 9999):
        raise ValueError("power_w must fit in PPPP (0000-9999)")
    if not (0 <= frequency_int <= 9999):
        raise ValueError("frequency_khz must fit in FFFF (0000-9999)")
    if not (0 <= pulse_width_int <= 9999):
        raise ValueError("pulse_width_us must fit in WWWW (0000-9999)")
    if not (0 <= detect_delay_int <= 99999999):
        raise ValueError("detect_delay_us must fit in DDDDDDDD (00000000-99999999)")

    data = (
        f"{program_id_int:02d}"
        f"{power_int:04d}"
        f"{frequency_int:04d}"
        f"{pulse_width_int:04d}"
        f"{detect_delay_int:08d}"
    )
    return compose_frame("40", data, checksum_fn=checksum_fn)

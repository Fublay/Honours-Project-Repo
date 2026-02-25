import re

from protocol.frame_codec import is_framed_command


DATA_RE = re.compile(
    r"t=([0-9.]+)\s+"
    r"y=([0-9.eE+-]+)"
    r"(?:\s+sp=([0-9.eE+-]+))?\s+"
    r"u=([0-9.eE+-]+)\s+"
    r"status=([A-Z]+)"
)
KV_RE = re.compile(r"([A-Za-z_]+)=([0-9.eE+-]+)")
DEBUG_B0_RE = re.compile(
    r"^\$B0"
    r"([0-9A-Fa-f]{8}):([0-9A-Fa-f]{8}):([0-9A-Fa-f]{8}):([0-9A-Fa-f]{8})"
    r"([0-9A-Fa-f]{2})$"
)


def parse_pid_reply(packet: str) -> dict:
    """
    Parse a GET_PID reply packet.
    Format: $B6<fields><checksum>\r\n
    """
    if not isinstance(packet, str):
        raise TypeError("packet must be a str")
    if not is_framed_command(packet):
        raise ValueError("Invalid packet format: expected frame starting with '$' and ending with '\\r\\n'")

    pkt = packet[:-2]
    if not pkt.startswith("$B6"):
        raise ValueError(f"Expected GET_PID reply ($B6...), got: {pkt[:10]}")
    if len(pkt) < 10:
        raise ValueError("Packet too short")

    received_checksum_str = pkt[-2:]
    # Keep exact spacing from device reply for checksum calculation.
    # Some controllers include leading spaces in checksum accumulation.
    data_portion = pkt[3:-2]
    calculated_checksum = sum(ord(ch) for ch in data_portion) % 256
    received_checksum = int(received_checksum_str, 16)
    if calculated_checksum != received_checksum:
        raise ValueError(
            f"Checksum mismatch: calculated {calculated_checksum:02X}, received {received_checksum_str}"
        )

    fields = [f.strip() for f in data_portion.split() if f.strip()]
    if len(fields) != 8:
        raise ValueError(f"Expected 8 PID parameter fields, got {len(fields)}")

    try:
        return {
            "pw_kp": float(fields[0]),
            "pw_ki": float(fields[1]),
            "pw_kd": float(fields[2]),
            "pp_kp": float(fields[3]),
            "pp_ki": float(fields[4]),
            "pp_kd": float(fields[5]),
            "holdoff": float(fields[6]),
            "sample_interval": float(fields[7]),
        }
    except ValueError as e:
        raise ValueError(f"Failed to parse PID values: {e}")


def parse_ack(line: str) -> tuple[bool, str]:
    """
    Parse controller ack like '*00'.
    Returns (is_success, code).
    """
    s = (line or "").strip()
    if not s.startswith("*") or len(s) < 3:
        return False, ""
    code = s[1:3]
    return code == "00", code


def parse_telemetry_line(line: str) -> dict | None:
    """
    Parse one telemetry line from any supported format:
      1) DATA t=... y=... u=... status=...
      2) key/value packet with power/period/width fields
      3) framed debug packet:
         $B0AAAAAAAA:BBBBBBBB:CCCCCCCC:DDDDDDDDXX
    """
    s = (line or "").strip()
    if not s:
        return None

    debug_match = DEBUG_B0_RE.match(s)
    if debug_match is not None:
        a_hex, b_hex, c_hex, d_hex, rx_checksum_hex = debug_match.groups()
        payload = s[3:-2]  # exact bytes after command id, before checksum
        calc_checksum = sum(ord(ch) for ch in payload) % 256
        rx_checksum = int(rx_checksum_hex, 16)
        if calc_checksum != rx_checksum:
            return None
        return {
            "t": None,
            "initial_power": float(int(a_hex, 16)),
            "current_power": float(int(b_hex, 16)),
            "pulse_width": float(int(c_hex, 16)),
            "pulse_period": float(int(d_hex, 16)),
            "status": "RUNNING",
        }

    match = DATA_RE.search(s) if s.startswith("DATA") else None
    if match is not None:
        t, y, _sp, u, status = match.groups()
        return {
            "t": float(t),
            "y": float(y),
            "u": float(u),
            "status": status,
        }

    # New packet style: key/value fields that include initial/current power
    # plus pulse period/width.
    kv = {k.strip().lower(): float(v) for k, v in KV_RE.findall(s)}

    def first(keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key in kv:
                return kv[key]
        return None

    initial_power = first(("initial_power", "initial", "init_power", "ip", "p0"))
    current_power = first(("current_power", "current", "cur_power", "cp", "p"))
    pulse_period = first(("pulse_period", "period", "pp"))
    pulse_width = first(("pulse_width", "width", "pw"))
    t_val = first(("t", "time", "time_s"))

    if None in (initial_power, current_power, pulse_period, pulse_width):
        return None

    return {
        "t": t_val,
        "initial_power": float(initial_power),
        "current_power": float(current_power),
        "pulse_period": float(pulse_period),
        "pulse_width": float(pulse_width),
        "status": "RUNNING",
    }

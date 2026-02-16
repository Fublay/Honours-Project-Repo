import re


def format_pid_value(value: float, field_width: int = 8) -> str:
    """
    Format a PID parameter value to a fixed-width string.
    
    Args:
        value: The numeric value to format
        field_width: Total field width (default 8 characters)
    
    Returns:
        Formatted string with leading/trailing spaces as needed
    """
    # Format as float with appropriate precision
    if abs(value) < 0.001:
        formatted = "0.0000"
    elif abs(value) < 10:
        formatted = f"{value:.4f}"
    elif abs(value) < 100:
        formatted = f"{value:.2f}"
    else:
        formatted = f"{value:.2f}"
    
    # Right-align to field_width
    return f"{formatted:>{field_width}}"


def parse_pid_reply(packet: str) -> dict:
    """
    Parse a GET_PID reply packet.
    
    Format: $B6  <PW_Kp>  <PW_Ki>  <PW_Kd>  <PP_Kp>  <PP_Ki>  <PP_Kd>  <Holdoff>  <SampleInterval><checksum>\r\n
    
    Args:
        packet: Raw reply packet string
    
    Returns:
        Dictionary with keys: pw_kp, pw_ki, pw_kd, pp_kp, pp_ki, pp_kd, holdoff, sample_interval
    
    Raises:
        ValueError on format problems or checksum mismatch
    """
    if not isinstance(packet, str):
        raise TypeError("packet must be a str")
    
    pkt = packet.strip()
    if not pkt.startswith("$B6"):
        raise ValueError(f"Expected GET_PID reply ($B6...), got: {pkt[:10]}")
    
    # Extract checksum (last 2 hex digits before \r\n)
    if len(pkt) < 10:
        raise ValueError("Packet too short")
    
    received_checksum_str = pkt[-2:]
    
    # Extract data portion (everything after $B6, excluding checksum)
    # Format: "  <8-char>  <8-char>  <8-char>  <8-char>  <8-char>  <8-char>  <8-char>  <8-char>"
    data_portion = pkt[3:-2].strip()
    
    # Verify checksum - checksum is calculated over the data portion only
    calculated_checksum = sum(ord(ch) for ch in data_portion) % 256
    received_checksum = int(received_checksum_str, 16)
    
    if calculated_checksum != received_checksum:
        raise ValueError(
            f"Checksum mismatch: calculated {calculated_checksum:02X}, received {received_checksum_str}"
        )
    
    # Split data portion into 8 fields (each 8 characters wide)
    # Handle variable spacing - split on whitespace and filter empty strings
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
    
    Format: $B5  <PW_Kp>  <PW_Ki>  <PW_Kd>  <PP_Kp>  <PP_Ki>  <PP_Kd>  <Holdoff>  <SampleInterval><checksum>\r\n
    
    Args:
        pw_kp, pw_ki, pw_kd: Pulse Width PID parameters (required)
        pp_kp, pp_ki, pp_kd: Pulse Period PID parameters (optional, uses current_values if None)
        holdoff: Holdoff period in ms (optional, uses current_values if None)
        sample_interval: Sample interval in ms (optional, uses current_values if None)
        current_values: Dictionary from parse_pid_reply() to use for unspecified parameters
        checksum_fn: Checksum function to use
    
    Returns:
        Bytes ready to write to serial
    """
    # Use current_values for unspecified parameters
    if current_values:
        pp_kp = pp_kp if pp_kp is not None else current_values.get("pp_kp", 0.15)
        pp_ki = pp_ki if pp_ki is not None else current_values.get("pp_ki", 0.14)
        pp_kd = pp_kd if pp_kd is not None else current_values.get("pp_kd", 0.05)
        holdoff = holdoff if holdoff is not None else current_values.get("holdoff", 400.0)
        sample_interval = sample_interval if sample_interval is not None else current_values.get("sample_interval", 300.0)
    else:
        # Default values if no current_values provided
        pp_kp = pp_kp if pp_kp is not None else 0.15
        pp_ki = pp_ki if pp_ki is not None else 0.14
        pp_kd = pp_kd if pp_kd is not None else 0.05
        holdoff = holdoff if holdoff is not None else 400.0
        sample_interval = sample_interval if sample_interval is not None else 300.0
    
    # Format all values to 8-character fields
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
    
    # Join with single space (protocol uses spaces between fields)
    data = " ".join(data_parts)
    
    # Compose frame using standard compose_frame function
    return compose_frame("B5", data, checksum_fn=checksum_fn)


def default_checksum_hex_2(_frame_without_checksum: bytes) -> str:
    """
    Placeholder checksum calculator.

    Protocol: $XXYYYYCC\\r\\n
      - XX: 2 hex digits command id
      - YYYY: command data (ASCII, variable length)
      - CC: 2 hex digits checksum (HEX)

    The actual checksum algorithm will be provided later.
    Must return exactly 2 hex digits (uppercase), e.g. "00", "7A".
    """
    if not isinstance(_frame_without_checksum, (bytes, bytearray)):
        raise TypeError("frame_without_checksum must be bytes")

    data_portion = _frame_without_checksum[3:] if len(_frame_without_checksum) > 3 else b""
    checksum_val = sum(data_portion) & 0xFF
    return f"{checksum_val:02X}"


def compose_frame(command_id_hex2: str, data: str, checksum_fn=default_checksum_hex_2) -> bytes:
    """
    Compose a framed command according to:
      $XXYYYYCC\\r\\n

    Args:
      command_id_hex2: Two hex digits command id (case-insensitive accepted)
      data: Command data payload (variable length). Any leading '$' and any CR/LF
            characters will be stripped.
      checksum_fn: callable(frame_without_checksum: bytes) -> str (2 hex digits)

    Returns:
      Bytes ready to write to serial.
    """
    cid = (command_id_hex2 or "").strip().upper()
    if not re.fullmatch(r"[0-9A-F]{2}", cid):
        raise ValueError(f"command_id_hex2 must be exactly 2 hex digits, got {command_id_hex2!r}")

    payload = (data or "").strip()
    if payload.startswith("$"):
        payload = payload[1:]
    payload = payload.replace("\r", "").replace("\n", "")

    frame_wo_checksum = f"${cid}{payload}".encode("ascii", errors="ignore")
    checksum = str(checksum_fn(frame_wo_checksum)).strip().upper()
    if not re.fullmatch(r"[0-9A-F]{2}", checksum):
        raise ValueError(f"checksum_fn must return exactly 2 hex digits, got {checksum!r}")

    return frame_wo_checksum + checksum.encode("ascii") + b"\r\n"


def parse_reply(packet: str) -> tuple[int, str]:
    """
    Parse and validate a reply frame, mirroring C++ ReplyParser::parseReply.

    Args:
      packet: Raw packet string, e.g. "$01DATA...CC\\r\\n" or "$FFCCCCDATA...CC\\r\\n".

    Returns:
      (cmd_id, accumulated_buffer) where:
        - cmd_id is the integer command id (normal or extended),
        - accumulated_buffer matches C++ ReplyParser's accumulatedBuffer:
            * normal command: payload only (YYYY),
            * extended command: 4-hex command id + payload.

    Raises:
      ValueError on format problems or checksum mismatch.
    """
    if not isinstance(packet, str):
        raise TypeError("packet must be a str")

    # Strip CR/LF and surrounding whitespace.
    pkt = packet.strip()
    if len(pkt) < 5 or not pkt.startswith("$"):
        raise ValueError("Invalid packet format")

    cmd_str = pkt[1:3]

    # Last 2 characters are the checksum hex.
    if len(pkt) < 5:
        raise ValueError("Packet too short for checksum")
    received_checksum_str = pkt[-2:]

    # Extract command id and accumulated buffer, following ReplyParser.
    if cmd_str == "FF":
        if len(pkt) < 10:
            raise ValueError("Invalid FF command format")

        cmd_hex4 = pkt[3:7]
        try:
            cmd_id = int(cmd_hex4, 16)
        except ValueError:
            raise ValueError("Invalid extended command ID") from None

        # accumulatedBuffer = cmd_str(4 hex) + payload
        payload = pkt[7:-2]
        accumulated = cmd_hex4 + payload
    else:
        try:
            cmd_id = int(cmd_str, 16)
        except ValueError:
            raise ValueError("Invalid command ID") from None

        # accumulatedBuffer = payload only
        accumulated = pkt[3:-2]

    # Compute checksum over accumulatedBuffer, like setChecksum().
    calculated = sum(ord(ch) for ch in accumulated) % 256

    try:
        received = int(received_checksum_str, 16)
    except ValueError:
        raise ValueError("Invalid checksum format") from None

    if calculated != received:
        raise ValueError(
            f"Checksum validation failed: expected {calculated}, got {received}"
        )

    return cmd_id, accumulated
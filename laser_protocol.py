import re


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
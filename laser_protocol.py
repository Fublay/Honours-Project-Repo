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
    return "00"


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


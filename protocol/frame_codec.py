"""Low-level frame encoding and decoding for the serial protocol.

The firmware protocol uses text frames with this shape:
    $<cmd><payload><checksum>\r\n
"""

import re


FRAME_RE = re.compile(r"^\$[^\r\n]*\r\n$")


def is_framed_command(packet: str) -> bool:
    """
    Returns True only for packets framed as "$...\\r\\n".
    """
    return isinstance(packet, str) and FRAME_RE.fullmatch(packet) is not None


def default_checksum_hex_2(frame_without_checksum: bytes) -> str:
    """Compute protocol checksum as 8-bit sum over payload characters."""
    if not isinstance(frame_without_checksum, (bytes, bytearray)):
        raise TypeError("frame_without_checksum must be bytes")

    s = frame_without_checksum.decode("ascii", errors="ignore")
    # Skip '$' + 2-byte command ID; checksum only uses data section here.
    data_portion = s[3:] if len(s) >= 3 else ""  # everything after '$' + 2-char cmd
    checksum = sum(ord(c) for c in data_portion) & 0xFF
    return f"{checksum:02X}"


def compose_frame(command_id_hex2: str, data: str, checksum_fn=default_checksum_hex_2) -> bytes:
    """
    Compose a framed command according to:
      $XXYYYYCC\\r\\n
    """
    # Normalize command ID and validate format before building a frame.
    cid = (command_id_hex2 or "").strip().upper()
    if not re.fullmatch(r"[0-9A-F]{2}", cid):
        raise ValueError(f"command_id_hex2 must be exactly 2 hex digits, got {command_id_hex2!r}")

    # Strip accidental framing/newline characters from payload input.
    payload = data or ""
    if payload.startswith("$"):
        payload = payload[1:]
    payload = payload.replace("\r", "").replace("\n", "")

    # Build checksum over "$<id><payload>" then append checksum + CRLF.
    frame_wo_checksum = f"${cid}{payload}".encode("ascii", errors="ignore")
    checksum = str(checksum_fn(frame_wo_checksum)).strip().upper()
    if not re.fullmatch(r"[0-9A-F]{2}", checksum):
        raise ValueError(f"checksum_fn must return exactly 2 hex digits, got {checksum!r}")

    return frame_wo_checksum + checksum.encode("ascii") + b"\r\n"


def parse_reply(packet: str) -> tuple[int, str]:
    """
    Parse and validate a reply frame.
    """
    if not isinstance(packet, str):
        raise TypeError("packet must be a str")
    if not is_framed_command(packet):
        raise ValueError("Invalid packet format: expected frame starting with '$' and ending with '\\r\\n'")

    # Remove CRLF; keep frame text for parsing.
    pkt = packet[:-2]
    if len(pkt) < 5:
        raise ValueError("Invalid packet format")

    cmd_str = pkt[1:3]
    received_checksum_str = pkt[-2:]

    # 'FF' uses an extended 4-hex command identifier format.
    if cmd_str == "FF":
        if len(pkt) < 10:
            raise ValueError("Invalid FF command format")
        cmd_hex4 = pkt[3:7]
        try:
            cmd_id = int(cmd_hex4, 16)
        except ValueError:
            raise ValueError("Invalid extended command ID") from None
        payload = pkt[7:-2]
        accumulated = cmd_hex4 + payload
    else:
        try:
            cmd_id = int(cmd_str, 16)
        except ValueError:
            raise ValueError("Invalid command ID") from None
        accumulated = pkt[3:-2]

    # Validate checksum against the payload section from this reply format.
    calculated = sum(ord(ch) for ch in accumulated) % 256
    try:
        received = int(received_checksum_str, 16)
    except ValueError:
        raise ValueError("Invalid checksum format") from None

    if calculated != received:
        raise ValueError(f"Checksum validation failed: expected {calculated}, got {received}")

    return cmd_id, accumulated

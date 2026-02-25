import re


FRAME_RE = re.compile(r"^\$[^\r\n]*\r\n$")


def is_framed_command(packet: str) -> bool:
    """
    Returns True only for packets framed as "$...\\r\\n".
    """
    return isinstance(packet, str) and FRAME_RE.fullmatch(packet) is not None


def default_checksum_hex_2(frame_without_checksum: bytes) -> str:
    if not isinstance(frame_without_checksum, (bytes, bytearray)):
        raise TypeError("frame_without_checksum must be bytes")

    s = frame_without_checksum.decode("ascii", errors="ignore")
    data_portion = s[3:] if len(s) >= 3 else ""  # everything after '$' + 2-char cmd
    checksum = sum(ord(c) for c in data_portion) & 0xFF
    return f"{checksum:02X}"


def compose_frame(command_id_hex2: str, data: str, checksum_fn=default_checksum_hex_2) -> bytes:
    """
    Compose a framed command according to:
      $XXYYYYCC\\r\\n
    """
    cid = (command_id_hex2 or "").strip().upper()
    if not re.fullmatch(r"[0-9A-F]{2}", cid):
        raise ValueError(f"command_id_hex2 must be exactly 2 hex digits, got {command_id_hex2!r}")

    payload = data or ""
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
    Parse and validate a reply frame.
    """
    if not isinstance(packet, str):
        raise TypeError("packet must be a str")
    if not is_framed_command(packet):
        raise ValueError("Invalid packet format: expected frame starting with '$' and ending with '\\r\\n'")

    pkt = packet[:-2]
    if len(pkt) < 5:
        raise ValueError("Invalid packet format")

    cmd_str = pkt[1:3]
    received_checksum_str = pkt[-2:]

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

    calculated = sum(ord(ch) for ch in accumulated) % 256
    try:
        received = int(received_checksum_str, 16)
    except ValueError:
        raise ValueError("Invalid checksum format") from None

    if calculated != received:
        raise ValueError(f"Checksum validation failed: expected {calculated}, got {received}")

    return cmd_id, accumulated

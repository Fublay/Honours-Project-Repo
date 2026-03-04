"""Map command IDs to readable names.

This lets logs/reporting show human-friendly names instead of raw hex IDs.
"""

import laser_command_ids as CMD


def _is_hex_command_id(value: str) -> bool:
    """Return True only for two-character hexadecimal IDs (for example 'B6')."""
    if not isinstance(value, str):
        return False
    v = value.strip().upper()
    if len(v) != 2:
        return False
    return all(ch in "0123456789ABCDEF" for ch in v)


def _build_command_name_by_id() -> dict[str, str]:
    """Scan `laser_command_ids.py` and build a reverse lookup table."""
    command_name_by_id: dict[str, str] = {}
    for name, value in vars(CMD).items():
        # Skip private attributes and non-command values.
        if name.startswith("_"):
            continue
        if _is_hex_command_id(value):
            command_name_by_id[value.strip().upper()] = name
    return command_name_by_id


COMMAND_NAME_BY_ID = _build_command_name_by_id()


def command_name(command_id_hex2: str) -> str:
    """Translate command ID to name; returns 'UNKNOWN' when not registered."""
    return COMMAND_NAME_BY_ID.get((command_id_hex2 or "").strip().upper(), "UNKNOWN")


def is_supported_command(command_id_hex2: str) -> bool:
    """Quick boolean check used by validation and debugging paths."""
    return (command_id_hex2 or "").strip().upper() in COMMAND_NAME_BY_ID

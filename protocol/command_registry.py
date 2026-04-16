"""Map command IDs to readable names.

This lets logs/reporting show human-friendly names instead of raw hex IDs.
"""

# Build the reverse lookup from `laser_command_ids.py` so there is only one
# place where command names and IDs have to stay in sync.

import laser_command_ids as CMD


# Accept only the exact two-byte hex form used by normal controller commands.
def _is_hex_command_id(value: str) -> bool:
    """Return True only for two-character hexadecimal IDs (for example 'B6')."""
    if not isinstance(value, str):
        return False
    v = value.strip().upper()
    if len(v) != 2:
        return False
    return all(ch in "0123456789ABCDEF" for ch in v)


# Reflect the public constants from `laser_command_ids.py` into a table that is
# cheap to query during logging and validation.
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


# Fall back to "UNKNOWN" rather than raising so logs can still show unexpected
# traffic without crashing the caller.
def command_name(command_id_hex2: str) -> str:
    """Translate command ID to name; returns 'UNKNOWN' when not registered."""
    return COMMAND_NAME_BY_ID.get((command_id_hex2 or "").strip().upper(), "UNKNOWN")


# Separate helper for callers that only need a yes/no check.
def is_supported_command(command_id_hex2: str) -> bool:
    """Quick boolean check used by validation and debugging paths."""
    return (command_id_hex2 or "").strip().upper() in COMMAND_NAME_BY_ID

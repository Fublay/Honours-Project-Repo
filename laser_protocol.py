"""Public protocol API surface for the rest of the application.

This module simply re-exports the most important helpers so callers can import
from one place instead of remembering several lower-level modules.
"""

from protocol.command_composer import compose_set_pid_command, format_pid_value
from protocol.frame_codec import (
    compose_frame,
    default_checksum_hex_2,
    is_framed_command,
    parse_reply,
)
from protocol.reply_parser import parse_pid_reply

__all__ = [
    # Command build helpers.
    "compose_set_pid_command",
    "format_pid_value",
    # Generic frame utilities.
    "compose_frame",
    "default_checksum_hex_2",
    "is_framed_command",
    "parse_reply",
    # Reply parser for PID reads.
    "parse_pid_reply",
]

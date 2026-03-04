"""Compatibility import for older code paths.

Historically callers imported `SerialLineIO` from this module. We keep this
small wrapper so existing imports continue working after refactors.
"""

from transport.serial_interface import SerialLineIO

__all__ = ["SerialLineIO"]

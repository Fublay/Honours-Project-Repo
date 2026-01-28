"""
Command ID registry for the laser serial protocol.

Protocol framing: $XXYYYYCC\\r\\n
Where:
  - XX is a 2-hex-digit command id
  - YYYY is command data (variable length, ASCII)
  - CC is 2-hex-digit checksum

Update these values to match your laser firmware documentation.
"""

# NOTE: These are placeholders. Replace with the real hex IDs.
PING = "01"
SET_PID = "10"
SET_SP = "11"
START = "12"


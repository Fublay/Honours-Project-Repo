"""
Command ID registry for the laser serial protocol.

Based on Power Feedback systems RS232 protocol.
"""

# PID control commands
GET_PID = "B6"  # Read current PID values
SET_PID = "B5"  # Write PID values

# Legacy/placeholder commands (may need updating based on actual protocol)
PING = "6A"
SET_SP = "11"
START = "12"


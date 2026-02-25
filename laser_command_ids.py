"""
Command ID registry for the laser serial protocol.

Based on Power Feedback systems RS232 protocol.
"""

# PID control commands
DEBUG = "B0"
GET_PID = "B6"  # Read current PID values
SET_PID = "B5"  # Write PID values

# Legacy/placeholder commands (may need updating based on actual protocol)
START = "54"
STOP = "55"

GET_FLOW = "6A"

RUN = "71"
STANDBY = "70"
SHUTTER_CONTROL = "76"
TRIGGER = "77"


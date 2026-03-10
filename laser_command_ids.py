"""
Command ID registry for the laser serial protocol.

Based on Power Feedback systems RS232 protocol.
"""

# PID configuration commands.
# These are used to read/write tuning parameters from the laser firmware.
DEBUG = "B0"
GET_PID = "B6"  # Read current PID values
SET_PID = "B5"  # Write PID values

# Laser process control commands.
# START/STOP operate the active firing window, while RUN/STANDBY set
# the higher-level operating mode.
START = "54"
STOP = "55"

# Flow/diagnostic command used by protocol testing.
GET_FLOW = "6A"

# Machine state and safety-related controls.
RUN = "71"
STANDBY = "70"
SHUTTER_CONTROL = "76"
TRIGGER = "77"

# Enable/disable debug output masks (for example B0 telemetry stream).
SET_DEBUG = "0F"

# Program setup command used to set power/frequency fields before firing.
SET_PROGRAM = "40"
GET_PROGRAM = "41"

# Laser PID Tuning Tool

This project is a serial-connected PID tuning tool for a laser system.
In plain terms, it talks to the controller, runs repeated test shots, watches
the live power telemetry, scores how well each PID setup behaves, and keeps
searching for a better set of gains.

It is set up for real lab work rather than a toy demo. The script can:

- read the controller's current program settings
- reset the PID values back to known defaults
- run a bootstrap phase to find a safe local tuning region
- optimise candidate PID values over multiple trials
- show a live monitor while the run is happening
- save CSV files so the run can be reviewed afterwards

## What The Main Script Does

The main entry point is `tune_pid.py`.

When you start it, the tool connects to the laser controller over serial,
offers a small launcher, and then lets you:

- start a tuning run
- reset the controller to default PID values
- open the saved power graph viewer
- quit cleanly

During a tuning run, the code applies PID candidates, starts repeated laser
tests, collects telemetry from the controller, scores each result, and keeps
track of the best candidate found so far.

## Project Layout

- `tune_pid.py` - main application flow and CLI
- `tuning/` - search, scoring, bootstrap, and trial logic
- `transport/` - serial communication layer
- `protocol/` - frame building and reply parsing
- `pipeline/` - telemetry collection
- `ui/` - live monitor and offline graphing tools
- `domain/` - telemetry value mapping helpers

## Requirements

This repo does not currently include a `requirements.txt`, so you will usually
need to install the main dependencies yourself.

Recommended packages:

```bash
pip install numpy pyserial scikit-learn scikit-optimize matplotlib
```

Notes:

- `numpy` and `pyserial` are needed for normal operation.
- `scikit-learn` is used for the surrogate-guided search phase.
- `scikit-optimize` is used for the optional BO-style refinement steps.
- `matplotlib` is needed for the saved trace graph viewer.
- `tkinter` is used for the launcher and live monitor; it is usually bundled
  with standard Python installs.

## Setup

### Windows

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy pyserial scikit-learn scikit-optimize matplotlib
```

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pyserial scikit-learn scikit-optimize matplotlib
```

## Running The Tool

### Windows

```powershell
py tune_pid.py --port COM10 --baud 115200
```

### Linux

```bash
python tune_pid.py --port /dev/ttyUSB0 --baud 115200
```

If you do not want the launcher GUI, run:

```bash
python tune_pid.py --port /dev/ttyUSB0 --baud 115200 --no-gui
```

Replace the port with whatever your controller is actually connected to.

## Useful Options

Some of the flags you are most likely to care about are:

- `--iters` - number of counted optimisation trials
- `--desired-output` - target power/output value
- `--frequency-khz` - startup program frequency sent to the controller
- `--test-duration-s` - duration of each individual test
- `--power-csv` - CSV file used by the graphing tool
- `--no-gui` - use console prompts instead of the launcher window

Example:

```bash
python tune_pid.py --port /dev/ttyUSB0 --baud 115200 --iters 30 --desired-output 525 --frequency-khz 0
```

## What Gets Saved

After a run, the tool writes CSV files into the project root:

- `tuning_history.csv` - trial-by-trial tuning results and summary metrics
- `tuning_trace_features.csv` - extracted features from the recorded traces
- `tuning_power_readings.csv` - time series power data from the tests

These files are useful if you want to compare runs, inspect the best candidate,
or reopen the saved traces in the graph tool later.

## A Typical Workflow

1. Connect the controller and confirm the serial port.
2. Start `tune_pid.py`.
3. Use the launcher to start a run, reset defaults, or inspect saved graphs.
4. Let the bootstrap phase find a stable starting region.
5. Let the optimiser work through the requested trials.
6. Review the live monitor during the run and the CSV outputs afterwards.

## A Few Practical Notes

- This tool is meant for real hardware, so make sure the controller is in the
  expected state before starting a run.
- If graphing fails, check that `matplotlib` is installed in the same virtual
  environment you are using to run the tuner.
- If the GUI is unavailable, the script falls back to console prompts.
- If you only want to restore the known-safe defaults, launch the tool and use
  the reset action instead of running a full optimisation pass.

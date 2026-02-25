def map_telemetry_values(raw: dict) -> dict:
    """
    Central place to map raw telemetry values to usable engineering values.

    Legacy DATA mode:
      uses y/u directly.

    Power packet mode:
      derives process_value from power * duty cycle, where
      duty_cycle = pulse_width / pulse_period.
    """
    if "initial_power" in raw and "current_power" in raw:
        pulse_period = float(raw["pulse_period"])
        pulse_width = float(raw["pulse_width"])
        duty_cycle = (pulse_width / pulse_period) if pulse_period > 0 else 0.0
        process_value = float(raw["current_power"]) * duty_cycle
        return {
            "time_s": float(raw["t"]) if raw.get("t") is not None else None,
            "process_value": process_value,
            "control_output": pulse_width,
            "status": str(raw.get("status", "RUNNING")),
            "initial_power": float(raw["initial_power"]),
            "current_power": float(raw["current_power"]),
            "pulse_period": pulse_period,
            "pulse_width": pulse_width,
            "duty_cycle": duty_cycle,
        }

    return {
        "time_s": float(raw["t"]),
        "process_value": float(raw["y"]),
        "control_output": float(raw["u"]),
        "status": str(raw["status"]),
    }

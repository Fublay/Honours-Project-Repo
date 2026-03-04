"""Convert parsed telemetry packets into a common shape used by the tuner.

The optimization/scoring code expects a single dictionary format no matter
which telemetry packet type the laser emitted.
"""


def map_telemetry_values(raw: dict) -> dict:
    """Normalize one telemetry sample into fields the rest of the app uses.

    Why this exists:
    - The controller can send multiple packet styles.
    - Later code should not care about packet style details.
    - This function gives everything a single, predictable format.
    """
    # B0 debug packets include explicit power and pulse information.
    if "initial_power" in raw and "current_power" in raw:
        pulse_period = float(raw["pulse_period"])
        pulse_width = float(raw["pulse_width"])
        # Duty cycle is "how much of each pulse period is ON".
        duty_cycle = (pulse_width / pulse_period) if pulse_period > 0 else 0.0
        # For power tuning, process value should be actual measured power.
        process_value = float(raw["current_power"])
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

    # Legacy "DATA ..." packets already expose process output directly as y/u.
    return {
        "time_s": float(raw["t"]),
        "process_value": float(raw["y"]),
        "control_output": float(raw["u"]),
        "status": str(raw["status"]),
    }

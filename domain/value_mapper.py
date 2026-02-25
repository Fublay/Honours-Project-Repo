def map_telemetry_values(raw: dict) -> dict:
    """
    Central place to map raw telemetry values to usable engineering values.
    Currently identity mapping with explicit keys, ready for scaling/offset rules.
    """
    return {
        "time_s": float(raw["t"]),
        "process_value": float(raw["y"]),
        "control_output": float(raw["u"]),
        "status": str(raw["status"]),
    }

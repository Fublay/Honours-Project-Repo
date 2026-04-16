"""Trace feature extraction and controller scoring helpers."""

from __future__ import annotations

import math

import numpy as np


# Small guard helpers keep the feature calculations from being cluttered with
# repeated empty-list handling.
def _safe_mean(values: list[float], default: float) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float(default)


# Same idea as `_safe_mean`, but for spread instead of centre.
def _safe_std(values: list[float], default: float) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(finite)) if finite else float(default)


# Trim obvious outliers before averaging repeat-level features.
def _robust_center(values: list[float], default: float) -> float:
    finite = sorted(float(v) for v in values if v is not None and np.isfinite(v))
    if not finite:
        return float(default)
    if len(finite) <= 2:
        return float(np.median(finite))
    if len(finite) >= 5:
        finite = finite[1:-1]
    return float(np.mean(finite))


# Clean and align time/value arrays before any metric extraction.
def _clean_trace(times: np.ndarray, readings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times_arr = np.asarray(times, dtype=float).flatten()
    readings_arr = np.asarray(readings, dtype=float).flatten()
    n = int(min(times_arr.size, readings_arr.size))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    times_arr = times_arr[:n]
    readings_arr = readings_arr[:n]
    mask = np.isfinite(times_arr) & np.isfinite(readings_arr)
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)
    times_arr = times_arr[mask]
    readings_arr = readings_arr[mask]
    if times_arr.size <= 1:
        return times_arr, readings_arr
    order = np.argsort(times_arr, kind="stable")
    return times_arr[order], readings_arr[order]


# Approximate per-sample dwell times for uneven telemetry spacing.
def _sample_widths(times: np.ndarray) -> np.ndarray:
    if times.size <= 1:
        return np.ones(int(max(1, times.size)), dtype=float)
    deltas = np.diff(times)
    positive = deltas[deltas > 1e-9]
    default_dt = float(np.median(positive)) if positive.size else 1.0
    widths = np.empty(times.size, dtype=float)
    widths[:-1] = np.where(deltas > 1e-9, deltas, default_dt)
    widths[-1] = default_dt
    return widths


# Trapezoidal integral helper shared by several error-area metrics.
def _integral(times: np.ndarray, values: np.ndarray) -> float:
    if times.size == 0 or values.size == 0:
        return 0.0
    if times.size == 1:
        return float(abs(values[0]))
    deltas = np.diff(times)
    if deltas.size == 0:
        return float(abs(values[0]))
    left = np.asarray(values[:-1], dtype=float)
    right = np.asarray(values[1:], dtype=float)
    area = 0.5 * (left + right) * deltas
    return float(np.sum(area))


# Return the first true index without forcing each caller to repeat the same
# flatnonzero boilerplate.
def _first_index(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    return int(idx[0])


# Rise time is measured from the first 10% crossing to the first 90% crossing
# of the total move toward the target.
def _compute_rise_time(times: np.ndarray, readings: np.ndarray, target: float) -> float | None:
    if times.size < 2 or readings.size < 2:
        return None
    start = float(readings[0])
    amplitude = float(target - start)
    if abs(amplitude) < 1e-9:
        return 0.0 if abs(start - target) < max(abs(target) * 0.05, 1.0) else None
    norm = (readings - start) / amplitude
    i10 = _first_index(norm >= 0.1)
    i90 = _first_index(norm >= 0.9)
    if i10 is None or i90 is None or i90 < i10:
        return None
    return float(times[i90] - times[i10])


# Settling only counts once the signal enters the tolerance band and mostly
# stays there for the rest of the trace.
def _compute_settling_time(
    times: np.ndarray,
    abs_error: np.ndarray,
    tolerance: float,
    *,
    settled_window_samples: int = 5,
    settle_success_ratio: float = 0.85,
) -> float | None:
    if times.size == 0 or abs_error.size == 0 or tolerance <= 0:
        return None
    within = abs_error <= tolerance
    window = max(2, int(min(settled_window_samples, max(2, times.size))))
    for start_idx in range(0, max(1, times.size - window + 1)):
        window_ok = within[start_idx : start_idx + window]
        if window_ok.size < window or not np.all(window_ok):
            continue
        tail = within[start_idx:]
        if float(np.mean(tail.astype(float))) >= float(settle_success_ratio):
            return float(times[start_idx])
    return None


# Count sign flips while ignoring tiny oscillations inside a deadband.
def _count_sign_changes(values: np.ndarray, deadband: float) -> int:
    signs = []
    for value in values:
        if abs(float(value)) <= deadband:
            continue
        signs.append(1 if value > 0 else -1)
    if len(signs) < 2:
        return 0
    return int(sum(1 for prev, curr in zip(signs, signs[1:]) if prev != curr))


# Keep a compact summary of the early part of the trace for any future
# early-stop modelling or heuristics.
def extract_early_trace_features(
    times: np.ndarray,
    readings: np.ndarray,
    desired_output: float,
    *,
    early_window_s: float = 2.0,
) -> dict:
    """Summarise the first part of a trace for future early-abort models."""
    times_arr, readings_arr = _clean_trace(times, readings)
    if times_arr.size == 0:
        return {
            "early_window_s": float(early_window_s),
            "early_sample_count": 0.0,
            "early_slope": 0.0,
            "early_mean_error": float(abs(desired_output)),
            "early_max_error": float(abs(desired_output)),
        }

    window_mask = times_arr <= (float(times_arr[0]) + float(max(early_window_s, 0.1)))
    early_times = times_arr[window_mask]
    early_readings = readings_arr[window_mask]
    if early_times.size < 2:
        early_times = times_arr[: min(3, times_arr.size)]
        early_readings = readings_arr[: early_times.size]
    early_errors = early_readings - float(desired_output)
    duration = float(max(early_times[-1] - early_times[0], 1e-6))
    early_slope = float((early_readings[-1] - early_readings[0]) / duration) if early_times.size >= 2 else 0.0
    return {
        "early_window_s": float(early_window_s),
        "early_sample_count": float(early_times.size),
        "early_slope": early_slope,
        "early_mean_error": float(np.mean(np.abs(early_errors))) if early_errors.size else 0.0,
        "early_max_error": float(np.max(np.abs(early_errors))) if early_errors.size else 0.0,
    }


# Extract the repeat-level features that feed both scoring and diagnostics.
def extract_repeat_features(
    times: np.ndarray,
    readings: np.ndarray,
    desired_output: float,
    *,
    settled_window_samples: int = 5,
    tolerance_pct: float = 0.05,
    hold_tail_fraction: float = 0.33,
    hold_min_samples: int = 5,
) -> dict:
    """Compute lightweight control features from one repeat trace."""
    times_arr, readings_arr = _clean_trace(times, readings)
    empty_result = {
        "start_error": 999.0,
        "track_error": 999.0,
        "deviation": 999.0,
        "max_error": 999.0,
        "overshoot_pct": 100.0,
        "settling_time_s": math.nan,
        "rise_time_s": math.nan,
        "steady_state_error": 999.0,
        "iae": 999.0,
        "ise": 999.0,
        "itae": 999.0,
        "peak_value": math.nan,
        "peak_time_s": math.nan,
        "time_in_tolerance_s": 0.0,
        "time_to_first_tolerance_s": math.nan,
        "post_settle_variance": 999.0,
        "early_slope": 0.0,
        "oscillation_count": 0.0,
        "area_above_target": 0.0,
        "area_below_target": 0.0,
        "trace_duration_s": 0.0,
        "hold_duration_s": 0.0,
        "hold_mean_error": 999.0,
        "hold_variance": 999.0,
        "hold_drift": 999.0,
        "hold_time_in_tolerance_ratio": 0.0,
        "hold_oscillation_count": 0.0,
        "hold_quality": 999.0,
    }
    if times_arr.size == 0:
        return empty_result

    # Core tracking metrics.
    target = float(desired_output)
    base = max(abs(target), 1e-6)
    tolerance = max(base * float(tolerance_pct), 1e-6)
    widths = _sample_widths(times_arr)
    error = readings_arr - target
    abs_error = np.abs(error)
    start_error = abs(float(readings_arr[0]) - target)
    track_error = float(np.mean(abs_error))
    deviation = float(np.std(readings_arr))
    max_error = float(np.max(abs_error))
    peak_idx = int(np.argmax(readings_arr))
    peak_value = float(readings_arr[peak_idx])
    peak_time_s = float(times_arr[peak_idx])
    overshoot_pct = max(0.0, (peak_value - target) / base * 100.0)
    settling_time_s = _compute_settling_time(
        times_arr,
        abs_error,
        tolerance,
        settled_window_samples=settled_window_samples,
    )
    rise_time_s = _compute_rise_time(times_arr, readings_arr, target)
    steady_slice = readings_arr[-max(3, readings_arr.size // 4) :]
    steady_state_error = float(np.mean(steady_slice) - target) if steady_slice.size else 999.0
    iae = _integral(times_arr, abs_error)
    ise = _integral(times_arr, error**2)
    itae = _integral(times_arr, np.abs(error) * np.maximum(times_arr - float(times_arr[0]), 0.0))
    within_tol = abs_error <= tolerance
    time_in_tolerance_s = float(np.sum(widths[within_tol])) if within_tol.size else 0.0
    first_tol_idx = _first_index(within_tol)
    time_to_first_tolerance_s = float(times_arr[first_tol_idx]) if first_tol_idx is not None else math.nan
    post_settle_variance = deviation
    if settling_time_s is not None:
        post_mask = times_arr >= settling_time_s
        if np.any(post_mask):
            post_settle_variance = float(np.var(readings_arr[post_mask]))
    early_features = extract_early_trace_features(times_arr, readings_arr, target)
    area_above_target = _integral(times_arr, np.maximum(error, 0.0))
    area_below_target = _integral(times_arr, np.maximum(-error, 0.0))
    oscillation_count = _count_sign_changes(error, deadband=0.03 * base)
    trace_duration_s = float(max(times_arr[-1] - times_arr[0], 0.0)) if times_arr.size >= 2 else 0.0

    # Hold metrics focus on the tail of the trace, where steady control matters
    # more than startup behavior.
    hold_sample_count = min(
        readings_arr.size,
        max(int(math.ceil(readings_arr.size * float(hold_tail_fraction))), int(hold_min_samples)),
    )
    hold_readings = readings_arr[-hold_sample_count:]
    hold_times = times_arr[-hold_sample_count:]
    hold_errors = hold_readings - target
    hold_widths = _sample_widths(hold_times)
    hold_duration_s = float(np.sum(hold_widths)) if hold_widths.size else 0.0
    hold_mean_error = float(np.mean(hold_readings) - target) if hold_readings.size else 999.0
    hold_variance = float(np.var(hold_readings)) if hold_readings.size else 999.0
    hold_midpoint = max(1, hold_readings.size // 2)
    hold_first = hold_readings[:hold_midpoint]
    hold_second = hold_readings[hold_midpoint:]
    if hold_first.size and hold_second.size:
        hold_drift = float(abs(np.mean(hold_second) - np.mean(hold_first)))
    else:
        hold_drift = float(abs(hold_mean_error))
    hold_within_tol = np.abs(hold_errors) <= tolerance
    if hold_widths.size and np.sum(hold_widths) > 0:
        hold_time_in_tolerance_ratio = float(np.sum(hold_widths[hold_within_tol]) / np.sum(hold_widths))
    elif hold_within_tol.size:
        hold_time_in_tolerance_ratio = float(np.mean(hold_within_tol))
    else:
        hold_time_in_tolerance_ratio = 0.0
    hold_oscillation_count = float(_count_sign_changes(hold_errors, deadband=0.02 * base))
    hold_mean_error_pct = 100.0 * abs(hold_mean_error) / base
    hold_std_pct = 100.0 * math.sqrt(max(hold_variance, 0.0)) / base
    hold_drift_pct = 100.0 * hold_drift / base
    hold_tolerance_miss_pct = 100.0 * max(0.0, 1.0 - hold_time_in_tolerance_ratio)
    hold_quality = (
        3.0 * hold_mean_error_pct
        + 2.0 * hold_tolerance_miss_pct
        + 2.0 * hold_std_pct
        + 1.5 * hold_drift_pct
        + 4.0 * hold_oscillation_count
    )
    return {
        "start_error": start_error,
        "track_error": track_error,
        "deviation": deviation,
        "max_error": max_error,
        "overshoot_pct": float(overshoot_pct),
        "settling_time_s": float(settling_time_s) if settling_time_s is not None else math.nan,
        "rise_time_s": float(rise_time_s) if rise_time_s is not None else math.nan,
        "steady_state_error": steady_state_error,
        "iae": float(iae),
        "ise": float(ise),
        "itae": float(itae),
        "peak_value": peak_value,
        "peak_time_s": peak_time_s,
        "time_in_tolerance_s": time_in_tolerance_s,
        "time_to_first_tolerance_s": time_to_first_tolerance_s,
        "post_settle_variance": float(post_settle_variance),
        "early_slope": float(early_features["early_slope"]),
        "oscillation_count": float(oscillation_count),
        "area_above_target": float(area_above_target),
        "area_below_target": float(area_below_target),
        "early_mean_error": float(early_features["early_mean_error"]),
        "early_max_error": float(early_features["early_max_error"]),
        "trace_duration_s": float(trace_duration_s),
        "hold_duration_s": float(hold_duration_s),
        "hold_mean_error": float(hold_mean_error),
        "hold_variance": float(hold_variance),
        "hold_drift": float(hold_drift),
        "hold_time_in_tolerance_ratio": float(hold_time_in_tolerance_ratio),
        "hold_oscillation_count": float(hold_oscillation_count),
        "hold_quality": float(hold_quality),
    }


# Collapse per-repeat features into one candidate-level feature vector.
def aggregate_repeat_features(repeat_features: list[dict]) -> dict:
    """Aggregate repeat-level trace features into candidate-level features."""
    if not repeat_features:
        return {
            "feature_early_slope_mean": 0.0,
            "feature_peak_value_mean": math.nan,
            "feature_peak_time_s_mean": math.nan,
            "feature_overshoot_pct_mean": 100.0,
            "feature_time_to_first_tolerance_s_mean": math.nan,
            "feature_time_in_tolerance_s_mean": 0.0,
            "feature_oscillation_count_mean": 0.0,
            "feature_area_above_target_mean": 0.0,
            "feature_area_below_target_mean": 0.0,
            "feature_post_settle_variance_mean": 999.0,
            "feature_early_mean_error_mean": 999.0,
            "feature_early_max_error_mean": 999.0,
            "feature_hold_duration_s_mean": 0.0,
            "feature_hold_mean_error_mean": 999.0,
            "feature_hold_variance_mean": 999.0,
            "feature_hold_drift_mean": 999.0,
            "feature_hold_time_in_tolerance_ratio_mean": 0.0,
            "feature_hold_oscillation_count_mean": 0.0,
            "feature_hold_quality_mean": 999.0,
            "feature_peak_value_std": 0.0,
            "feature_overshoot_pct_std": 0.0,
        }

    def values(key: str) -> list[float]:
        return [float(item[key]) for item in repeat_features if key in item and np.isfinite(float(item[key]))]

    return {
        "feature_early_slope_mean": _robust_center(values("early_slope"), 0.0),
        "feature_peak_value_mean": _robust_center(values("peak_value"), math.nan),
        "feature_peak_time_s_mean": _robust_center(values("peak_time_s"), math.nan),
        "feature_overshoot_pct_mean": _robust_center(values("overshoot_pct"), 100.0),
        "feature_time_to_first_tolerance_s_mean": _robust_center(values("time_to_first_tolerance_s"), math.nan),
        "feature_time_in_tolerance_s_mean": _robust_center(values("time_in_tolerance_s"), 0.0),
        "feature_oscillation_count_mean": _robust_center(values("oscillation_count"), 0.0),
        "feature_area_above_target_mean": _robust_center(values("area_above_target"), 0.0),
        "feature_area_below_target_mean": _robust_center(values("area_below_target"), 0.0),
        "feature_post_settle_variance_mean": _robust_center(values("post_settle_variance"), 999.0),
        "feature_early_mean_error_mean": _robust_center(values("early_mean_error"), 999.0),
        "feature_early_max_error_mean": _robust_center(values("early_max_error"), 999.0),
        "feature_hold_duration_s_mean": _robust_center(values("hold_duration_s"), 0.0),
        "feature_hold_mean_error_mean": _robust_center(values("hold_mean_error"), 999.0),
        "feature_hold_variance_mean": _robust_center(values("hold_variance"), 999.0),
        "feature_hold_drift_mean": _robust_center(values("hold_drift"), 999.0),
        "feature_hold_time_in_tolerance_ratio_mean": _robust_center(values("hold_time_in_tolerance_ratio"), 0.0),
        "feature_hold_oscillation_count_mean": _robust_center(values("hold_oscillation_count"), 0.0),
        "feature_hold_quality_mean": _robust_center(values("hold_quality"), 999.0),
        "feature_peak_value_std": _safe_std(values("peak_value"), 0.0),
        "feature_overshoot_pct_std": _safe_std(values("overshoot_pct"), 0.0),
    }


# Combine repeat traces and repeat metadata into the aggregate metrics used by
# scoring, logging, and bootstrap safety checks.
def compute_trial_metrics(
    per_test_powers: list[np.ndarray],
    per_test_times: list[np.ndarray],
    per_test_meta: list[dict],
    desired_output: float,
    *,
    settled_window_samples: int = 5,
):
    """Convert repeated traces into aggregate control metrics and features."""
    repeat_features: list[dict] = []
    strict_bad_rates: list[float] = []
    oscillation_rates: list[float] = []
    invalid_flags: list[float] = []
    per_test_scores_unweighted: list[float] = []

    for readings, times, meta in zip(per_test_powers, per_test_times, per_test_meta):
        # Feature extraction works from the raw traces, while the metadata adds
        # trial-runner decisions like invalidation and oscillation strikes.
        features = extract_repeat_features(
            times,
            readings,
            desired_output,
            settled_window_samples=settled_window_samples,
        )
        repeat_features.append(features)
        strict_bad_rates.append(float(meta.get("strict_bad_rate", 1.0)))
        oscillation_rates.append(float(meta.get("oscillation_rate", 1.0)))
        invalid_flags.append(1.0 if bool(meta.get("invalid", False)) else 0.0)
        per_test_scores_unweighted.append(
            features["track_error"]
            + features["max_error"]
            + features["overshoot_pct"]
            + abs(features["steady_state_error"])
            + features["iae"]
            + strict_bad_rates[-1]
            + oscillation_rates[-1]
        )

    metrics = {
        "start_error": _robust_center([f["start_error"] for f in repeat_features], 999.0),
        "track_error": _robust_center([f["track_error"] for f in repeat_features], 999.0),
        "deviation": _robust_center([f["deviation"] for f in repeat_features], 999.0),
        "max_error": _robust_center([f["max_error"] for f in repeat_features], 999.0),
        "overshoot_pct": _robust_center([f["overshoot_pct"] for f in repeat_features], 100.0),
        "settling_time_s": _robust_center([f["settling_time_s"] for f in repeat_features], 999.0),
        "rise_time_s": _robust_center([f["rise_time_s"] for f in repeat_features], 999.0),
        "steady_state_error": _robust_center([f["steady_state_error"] for f in repeat_features], 999.0),
        "iae": _robust_center([f["iae"] for f in repeat_features], 999.0),
        "ise": _robust_center([f["ise"] for f in repeat_features], 999.0),
        "itae": _robust_center([f["itae"] for f in repeat_features], 999.0),
        "peak_value": _robust_center([f["peak_value"] for f in repeat_features], math.nan),
        "peak_time_s": _robust_center([f["peak_time_s"] for f in repeat_features], 999.0),
        "time_in_tolerance_s": _robust_center([f["time_in_tolerance_s"] for f in repeat_features], 0.0),
        "time_to_first_tolerance_s": _robust_center([f["time_to_first_tolerance_s"] for f in repeat_features], 999.0),
        "post_settle_variance": _robust_center([f["post_settle_variance"] for f in repeat_features], 999.0),
        "trace_duration_s": _robust_center([f["trace_duration_s"] for f in repeat_features], 0.0),
        "hold_duration_s": _robust_center([f["hold_duration_s"] for f in repeat_features], 0.0),
        "hold_mean_error": _robust_center([f["hold_mean_error"] for f in repeat_features], 999.0),
        "hold_variance": _robust_center([f["hold_variance"] for f in repeat_features], 999.0),
        "hold_drift": _robust_center([f["hold_drift"] for f in repeat_features], 999.0),
        "hold_time_in_tolerance_ratio": _robust_center([f["hold_time_in_tolerance_ratio"] for f in repeat_features], 0.0),
        "hold_oscillation_count": _robust_center([f["hold_oscillation_count"] for f in repeat_features], 0.0),
        "hold_quality": _robust_center([f["hold_quality"] for f in repeat_features], 999.0),
        "strict_bad_rate": _robust_center(strict_bad_rates, 1.0),
        "oscillation_rate": _robust_center(oscillation_rates, 1.0),
        "invalid_ratio": _safe_mean(invalid_flags, 1.0),
        "repeatability": _safe_std(per_test_scores_unweighted, 999.0),
    }
    metrics.update(aggregate_repeat_features(repeat_features))
    return metrics, repeat_features


# Cheap repeat-level score used during a candidate run before the full weighted
# controller score is computed.
def score_single_repeat(readings: np.ndarray, meta: dict, desired_output: float) -> float:
    """Score one repeat quickly so unstable candidates can be stopped early."""
    if readings.size == 0:
        return 999.0

    start_power = float(readings[0])
    abs_error = np.abs(readings - desired_output)
    start_error = abs(start_power - desired_output)
    track_error = float(np.mean(abs_error))
    deviation = float(np.std(readings))
    max_error = float(np.max(abs_error))
    strict_bad_rate = float(meta.get("strict_bad_rate", 1.0))
    oscillation_rate = float(meta.get("oscillation_rate", 1.0))
    hold_error = float(np.mean(readings[-max(5, readings.size // 3) :]) - desired_output)
    hold_variance = float(np.var(readings[-max(5, readings.size // 3) :]))
    return (
        0.25 * start_error
        + 0.50 * track_error
        + 0.75 * deviation
        + 0.75 * max_error
        + 2.0 * abs(hold_error)
        + 1.5 * hold_variance
        + 3.0 * strict_bad_rate
        + 4.0 * oscillation_rate
    )


# Final scalar objective used by the tuner.
# Lower is better, with extra penalties layered on top for unsafe or unstable
# behaviour that should dominate small improvements elsewhere.
def score_controller(
    metrics: dict,
    *,
    w_start: float,
    w_track: float,
    w_dev: float,
    w_max: float,
    w_repeat: float,
    w_strict: float,
    w_osc: float,
    w_overshoot: float,
    w_settle: float,
    w_rise: float,
    w_steady: float,
    w_iae: float,
    w_ise: float,
    w_tolerance_time: float,
    w_post_var: float,
    w_hold: float,
    invalid_penalty: float,
    cancelled_candidate: bool,
    aborted: bool,
):
    """Combine control metrics into one scalar score (lower is better)."""
    hold_ratio = float(metrics.get("hold_time_in_tolerance_ratio", 0.0))
    # Discount average tracking error when the controller demonstrably holds the
    # target well in the tail.
    track_discount = 0.25 if hold_ratio >= 0.80 else (0.50 if hold_ratio >= 0.50 else 1.0)
    score = (
        w_start * float(metrics["start_error"])
        + w_track * float(metrics["track_error"]) * track_discount
        + w_dev * float(metrics["deviation"])
        + w_max * float(metrics["max_error"])
        + w_repeat * float(metrics["repeatability"])
        + w_strict * float(metrics["strict_bad_rate"])
        + w_osc * float(metrics["oscillation_rate"])
        + w_overshoot * float(metrics["overshoot_pct"])
        + w_settle * float(metrics["settling_time_s"])
        + w_rise * float(metrics["rise_time_s"])
        + w_steady * abs(float(metrics["steady_state_error"]))
        + w_iae * float(metrics["iae"])
        + w_ise * float(metrics["ise"])
        + w_post_var * float(metrics["post_settle_variance"])
        + w_hold * float(metrics.get("hold_quality", 999.0))
        - w_tolerance_time * float(metrics["time_in_tolerance_s"])
        + invalid_penalty * float(metrics["invalid_ratio"])
    )
    # Hard behavioural penalties sit outside the weighted sum because they
    # represent conditions the optimiser should avoid aggressively.
    if hold_ratio < 0.60:
        score += (0.60 - hold_ratio) * 300.0
    if float(metrics.get("hold_oscillation_count", 0.0)) >= 2.0:
        score += 120.0
    if float(metrics.get("hold_drift", 0.0)) > max(abs(float(metrics.get("steady_state_error", 0.0))), 1.0):
        score += 80.0
    if cancelled_candidate:
        score += 250.0
    if aborted:
        score += 500.0
    if float(metrics.get("invalid_ratio", 0.0)) >= 0.5:
        score += 200.0
    if float(metrics.get("oscillation_rate", 0.0)) >= 0.5:
        score += 150.0
    return float(score)

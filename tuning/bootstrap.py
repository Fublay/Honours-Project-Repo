"""Bootstrap readiness and safe-region assessment helpers."""

from __future__ import annotations

import math

import numpy as np

from tuning.search import propose_coordinate_candidate


# Keep axis ordering in one shared tuple so diagnostics and search code stay in
# sync whenever they refer to Kp/Ki/Kd by index.
AXIS_NAMES = ("kp", "ki", "kd")


# Bootstrap probes often revisit the same point.
# Collapse duplicates before counting spread or density so retries do not make a
# region look better explored than it really is.
def _dedupe_points(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Collapse repeated probes of the same PID tuple into a single point."""
    unique: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        key = tuple(round(float(value), 6) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tuple(float(value) for value in point))
    return unique


# "Safe" means the candidate completed cleanly enough to be used when deciding
# whether optimisation is allowed to begin.
def candidate_is_safe(
    metrics: dict,
    *,
    cancelled_candidate: bool,
    aborted: bool,
    max_invalid_ratio: float,
    max_oscillation_rate: float,
) -> bool:
    """Gate optimisation until bootstrap has produced stable candidates."""
    if cancelled_candidate or aborted:
        return False
    if float(metrics.get("invalid_ratio", 1.0)) > max_invalid_ratio:
        return False
    if float(metrics.get("oscillation_rate", 1.0)) > max_oscillation_rate:
        return False
    return True


# "Good" is stricter than "safe": still stable, but also competitive enough to
# treat as a promising point near the baseline.
def candidate_is_good(
    metrics: dict,
    score: float,
    *,
    cancelled_candidate: bool,
    aborted: bool,
    baseline_score: float | None,
    max_invalid_ratio: float,
    max_oscillation_rate: float,
    max_score_factor: float,
) -> bool:
    """Require stability plus a score near or better than the baseline."""
    if not candidate_is_safe(
        metrics,
        cancelled_candidate=cancelled_candidate,
        aborted=aborted,
        max_invalid_ratio=max_invalid_ratio,
        max_oscillation_rate=max_oscillation_rate,
    ):
        return False
    if baseline_score is None or baseline_score <= 0:
        return True
    return float(score) <= (float(baseline_score) * float(max_score_factor))


# Build per-axis diagnostics so the GUI can show whether bootstrap has really
# explored a usable local neighbourhood yet.
def compute_bootstrap_axis_status(
    safe_points: list[tuple[float, float, float]],
    *,
    min_points_per_axis: int,
    min_span_kp: float,
    min_span_ki: float,
    min_span_kd: float,
) -> list[dict]:
    """Secondary diagnostic view of bootstrap spread on each PID axis."""
    required_spans = (float(min_span_kp), float(min_span_ki), float(min_span_kd))
    unique_safe_points = _dedupe_points(safe_points)
    if not unique_safe_points:
        # Keep the empty-state payload fully shaped so the UI does not have to
        # special-case missing keys.
        return [
            {
                "axis_index": idx,
                "axis_name": AXIS_NAMES[idx],
                "distinct_safe_values": 0,
                "safe_span": 0.0,
                "required_distinct_values": int(min_points_per_axis),
                "required_safe_span": float(required_spans[idx]),
                "distinct_deficit": int(min_points_per_axis),
                "span_deficit": float(required_spans[idx]),
                "distinct_coverage": 0.0,
                "span_coverage": 0.0,
                "deficit_score": float(int(min_points_per_axis) + 1.0),
                "coverage_ratio": 0.0,
                "bootstrap_complete": False,
                "complete": False,
            }
            for idx in range(3)
        ]

    safe_arr = np.asarray(unique_safe_points, dtype=float)
    statuses: list[dict] = []
    for idx, required_span in enumerate(required_spans):
        # Coverage is tracked both by distinct tested values and by total span
        # because either one alone can be misleading.
        axis_values = np.round(safe_arr[:, idx], 6)
        distinct_safe_values = int(len(np.unique(axis_values)))
        safe_span = float(np.max(safe_arr[:, idx]) - np.min(safe_arr[:, idx])) if safe_arr.size else 0.0
        distinct_deficit = max(0, int(min_points_per_axis) - distinct_safe_values)
        span_deficit = max(0.0, float(required_span) - safe_span)
        distinct_ratio = (
            min(1.0, distinct_safe_values / float(max(1, int(min_points_per_axis))))
            if int(min_points_per_axis) > 0
            else 1.0
        )
        span_ratio = min(1.0, safe_span / float(max(required_span, 1e-9))) if required_span > 0 else 1.0
        coverage_ratio = min(distinct_ratio, span_ratio)
        bootstrap_complete = bool(distinct_deficit == 0 and span_deficit <= 1e-9)
        deficit_score = float(distinct_deficit) + (float(span_deficit) / float(max(required_span, 1e-9)))
        statuses.append(
            {
                "axis_index": idx,
                "axis_name": AXIS_NAMES[idx],
                "distinct_safe_values": distinct_safe_values,
                "safe_span": safe_span,
                "required_distinct_values": int(min_points_per_axis),
                "required_safe_span": float(required_span),
                "distinct_deficit": distinct_deficit,
                "span_deficit": span_deficit,
                "distinct_coverage": float(distinct_ratio),
                "span_coverage": float(span_ratio),
                "deficit_score": float(deficit_score),
                "coverage_ratio": float(coverage_ratio),
                "bootstrap_complete": bootstrap_complete,
                "complete": bootstrap_complete,
            }
        )
    return statuses


# Scale the "local neighbourhood" around the best point from both the current
# step sizes and the minimum spread targets.
def _local_radius(
    *,
    step_kp: float,
    step_ki: float,
    step_kd: float,
    min_span_kp: float,
    min_span_ki: float,
    min_span_kd: float,
) -> tuple[float, float, float]:
    return (
        max(float(step_kp) * 1.5, float(min_span_kp), 1e-3),
        max(float(step_ki) * 1.5, float(min_span_ki), 1e-3),
        max(float(step_kd) * 1.5, float(min_span_kd), 1e-4),
    )


# Keep only points that lie inside the current local neighbourhood box.
def _nearby_points(
    points: list[tuple[float, float, float]],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    if not points:
        return []
    pts = np.asarray(points, dtype=float)
    ctr = np.asarray(center, dtype=float)
    rad = np.asarray(radii, dtype=float)
    normalized = np.abs((pts - ctr) / rad)
    mask = np.max(normalized, axis=1) <= 1.0
    return [tuple(float(v) for v in row) for row in pts[mask]]


# Count how much local movement each axis has actually seen near the current
# best point.
def _axis_variation(points: list[tuple[float, float, float]], radii: tuple[float, float, float]) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[int, int, int]]:
    if not points:
        return (0, 0, 0), (0.0, 0.0, 0.0), (0, 0, 0)
    arr = np.asarray(points, dtype=float)
    unique_counts = tuple(int(len(np.unique(np.round(arr[:, idx], 6)))) for idx in range(3))
    spans = tuple(float(np.max(arr[:, idx]) - np.min(arr[:, idx])) for idx in range(3))
    varied_axes = tuple(1 if spans[idx] >= max(radii[idx] * 0.35, 1e-6) else 0 for idx in range(3))
    return unique_counts, spans, varied_axes


# Decide whether bootstrap has learned enough about the stable region around the
# current best PID to hand control over to the optimisation phase.
def assess_local_safe_region(
    safe_points: list[tuple[float, float, float]],
    good_points: list[tuple[float, float, float]],
    unsafe_points: list[tuple[float, float, float]],
    *,
    best_pid: tuple[float, float, float] | None,
    min_safe_candidates: int,
    min_good_candidates: int,
    min_points_per_axis: int,
    min_span_kp: float,
    min_span_ki: float,
    min_span_kd: float,
    step_kp: float,
    step_ki: float,
    step_kd: float,
) -> dict:
    """Decide whether bootstrap has found a usable local safe region."""
    unique_safe_points = _dedupe_points(safe_points)
    unique_good_points = _dedupe_points(good_points)
    unique_unsafe_points = _dedupe_points(unsafe_points)
    axis_statuses = compute_bootstrap_axis_status(
        unique_safe_points,
        min_points_per_axis=min_points_per_axis,
        min_span_kp=min_span_kp,
        min_span_ki=min_span_ki,
        min_span_kd=min_span_kd,
    )
    if best_pid is None:
        # Without a best point there is no local region to judge yet.
        return {
            "ready": False,
            "reason": "waiting for a best candidate",
            "axis_statuses": axis_statuses,
            "blocking_axis": None,
            "center_pid": None,
            "local_safe_count": 0,
            "local_good_count": 0,
            "local_unsafe_count": 0,
            "local_unique_counts": (0, 0, 0),
            "local_spans": (0.0, 0.0, 0.0),
            "local_variation_axes": 0,
            "radii": _local_radius(
                step_kp=step_kp,
                step_ki=step_ki,
                step_kd=step_kd,
                min_span_kp=min_span_kp,
                min_span_ki=min_span_ki,
                min_span_kd=min_span_kd,
            ),
            "global_safe_count": len(unique_safe_points),
            "global_good_count": len(unique_good_points),
        }

    radii = _local_radius(
        step_kp=step_kp,
        step_ki=step_ki,
        step_kd=step_kd,
        min_span_kp=min_span_kp,
        min_span_ki=min_span_ki,
        min_span_kd=min_span_kd,
    )
    local_safe = _nearby_points(unique_safe_points, best_pid, radii)
    local_good = _nearby_points(unique_good_points, best_pid, radii)
    local_safe_keys = {tuple(round(float(value), 6) for value in point) for point in local_safe}
    local_good_keys = {tuple(round(float(value), 6) for value in point) for point in local_good}
    local_unsafe = [
        point
        for point in _nearby_points(unique_unsafe_points, best_pid, radii)
        if tuple(round(float(value), 6) for value in point) not in local_safe_keys | local_good_keys
    ]
    local_unique_counts, local_spans, varied_axes = _axis_variation(local_safe, radii)
    local_variation_axes = int(sum(varied_axes))

    local_safe_target = max(3, min(int(min_safe_candidates), 4))
    if len(unique_safe_points) < int(min_safe_candidates):
        # Global safe count is still the first gate before local structure
        # matters.
        reason = f"only {len(unique_safe_points)} safe candidates collected so far"
        ready = False
    elif len(local_safe) < local_safe_target:
        reason = (
            f"local safe region not dense enough near best candidate "
            f"({len(local_safe)}/{local_safe_target} nearby safe points)"
        )
        ready = False
    elif local_variation_axes < 2:
        varied_names = [AXIS_NAMES[idx].upper() for idx, varied in enumerate(varied_axes) if varied]
        reason = (
            "local neighbourhood still too narrow around best candidate "
            f"(variation on {', '.join(varied_names) if varied_names else 'no axes'})"
        )
        ready = False
    elif len(local_unsafe) > max(1, len(local_safe) // 2):
        reason = (
            f"local instability cliff still present near best candidate "
            f"({len(local_unsafe)} nearby unstable point{'s' if len(local_unsafe) != 1 else ''})"
        )
        ready = False
    else:
        reason = "Local safe region found near best candidate"
        ready = True

    weakest_axis = None
    if not ready:
        # Surface the weakest axis so the UI and logs can explain what is still
        # missing from the local region.
        weakness = [
            (
                idx,
                float(local_spans[idx]) / float(max(radii[idx], 1e-9)),
                int(local_unique_counts[idx]),
            )
            for idx in range(3)
        ]
        weakness.sort(key=lambda item: (item[1], item[2], item[0]))
        weakest_axis = AXIS_NAMES[weakness[0][0]].upper() if weakness else None

    return {
        "ready": ready,
        "reason": reason,
        "axis_statuses": axis_statuses,
        "blocking_axis": weakest_axis,
        "center_pid": tuple(float(v) for v in best_pid),
        "local_safe_count": len(local_safe),
        "local_good_count": len(local_good),
        "local_unsafe_count": len(local_unsafe),
        "local_unique_counts": local_unique_counts,
        "local_spans": local_spans,
        "local_variation_axes": local_variation_axes,
        "radii": radii,
        "global_safe_count": len(unique_safe_points),
        "global_good_count": len(unique_good_points),
    }


# Bootstrap only finishes once both the hard minimum trial count and the local
# safe-region checks agree that the search has enough footing.
def assess_bootstrap_progress(
    *,
    bootstrap_trials_done: int,
    bootstrap_trials_minimum: int,
    max_bootstrap_trials: int,
    region_status: dict,
) -> dict:
    """Combine the bootstrap floor with local safe-region readiness."""
    minimum_required = max(1, int(bootstrap_trials_minimum))
    hard_cap = max(minimum_required, int(max_bootstrap_trials))
    minimum_reached = int(bootstrap_trials_done) >= minimum_required
    region_ready = bool(region_status.get("ready", False))

    if minimum_reached and region_ready:
        reason = "Bootstrap complete: switching to optimisation"
        ready = True
    elif not minimum_reached:
        reason = f"bootstrap floor not reached yet ({int(bootstrap_trials_done)}/{minimum_required})"
        ready = False
    else:
        reason = str(region_status.get("reason", "searching for stable local region"))
        ready = False

    return {
        "ready": ready,
        "reason": reason,
        "minimum_required": minimum_required,
        "minimum_reached": minimum_reached,
        "hard_cap": hard_cap,
        "blocking_axis": region_status.get("blocking_axis"),
        "region_ready": region_ready,
    }


# Pick the next bootstrap axis by favouring whichever axis is least explored
# near the current base point, while still respecting controller bounds.
def choose_bootstrap_axis(
    base_pid: tuple[float, float, float],
    *,
    safe_points: list[tuple[float, float, float]],
    axis_directions: list[float],
    preferred_axis_index: int,
    step_kp: float,
    step_ki: float,
    step_kd: float,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    min_points_per_axis: int,
    min_span_kp: float,
    min_span_ki: float,
    min_span_kd: float,
) -> tuple[int, list[dict]]:
    """Prefer probing axes that improve local-region confidence near the best point."""
    axis_statuses = compute_bootstrap_axis_status(
        safe_points,
        min_points_per_axis=min_points_per_axis,
        min_span_kp=min_span_kp,
        min_span_ki=min_span_ki,
        min_span_kd=min_span_kd,
    )
    radii = _local_radius(
        step_kp=step_kp,
        step_ki=step_ki,
        step_kd=step_kd,
        min_span_kp=min_span_kp,
        min_span_ki=min_span_ki,
        min_span_kd=min_span_kd,
    )
    local_safe = _nearby_points(safe_points, base_pid, radii)
    local_unique_counts, local_spans, _ = _axis_variation(local_safe, radii)

    ranked: list[tuple[tuple[float, ...], int]] = []
    for status in axis_statuses:
        axis_idx = int(status["axis_index"])
        # Reuse the coordinate-step proposal logic here so axis selection only
        # considers moves that are actually legal under the current bounds.
        _, _, _, actual_delta = propose_coordinate_candidate(
            base_pid,
            axis_index=axis_idx,
            axis_direction=axis_directions[axis_idx % 3],
            step_kp=step_kp,
            step_ki=step_ki,
            step_kd=step_kd,
            kp_max=kp_max,
            ki_max=ki_max,
            kd_max=kd_max,
        )
        if abs(float(actual_delta)) <= 1e-9:
            continue
        local_span_ratio = float(local_spans[axis_idx]) / float(max(radii[axis_idx], 1e-9))
        local_unique = int(local_unique_counts[axis_idx])
        global_coverage = float(status.get("coverage_ratio", 0.0))
        ranked.append(
            (
                (
                    local_span_ratio,
                    float(local_unique),
                    global_coverage,
                    0.0 if axis_idx == (int(preferred_axis_index) % 3) else 1.0,
                    float(axis_idx),
                ),
                axis_idx,
            )
        )

    if not ranked:
        return int(preferred_axis_index) % 3, axis_statuses

    # Lowest tuple wins: least local span, least local uniqueness, then least
    # global coverage.
    ranked.sort(key=lambda item: item[0])
    return int(ranked[0][1]), axis_statuses

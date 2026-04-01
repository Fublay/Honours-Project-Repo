"""Candidate proposal helpers for bootstrap and optimisation."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

try:
    from skopt.space import Real
except Exception:
    Real = None

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    HAVE_SKLEARN = True
except Exception:
    ExtraTreesRegressor = None
    RandomForestRegressor = None
    HAVE_SKLEARN = False


SURROGATE_FEATURE_NAMES = ("kp", "ki", "kd", "desired_output", "frequency_khz")
BayesSpaceDim = Any
PidBounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


class OnlineSurrogateModel:
    """Small refit-on-update surrogate wrapper for embedded-friendly tuning."""

    def __init__(self, model_name: str, *, random_state: int = 42):
        self.model_name = str(model_name)
        self.random_state = int(random_state)
        self.model = None
        self.is_available = False
        self.last_error = ""
        self.last_fit_count = 0

    def _build_model(self):
        if not HAVE_SKLEARN or self.model_name == "none":
            raise RuntimeError("scikit-learn unavailable")
        if self.model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=48,
                max_depth=10,
                min_samples_leaf=2,
                random_state=self.random_state,
            )
        return ExtraTreesRegressor(
            n_estimators=48,
            max_depth=10,
            min_samples_leaf=2,
            random_state=self.random_state,
        )

    def fit(self, rows: list[dict], *, min_samples: int) -> bool:
        usable = [row for row in rows if np.isfinite(float(row.get("score", math.inf)))]
        if len(usable) < int(min_samples):
            self.is_available = False
            self.model = None
            self.last_fit_count = len(usable)
            return False
        try:
            x = np.asarray(
                [[float(row[name]) for name in SURROGATE_FEATURE_NAMES] for row in usable],
                dtype=float,
            )
            y = np.asarray([float(row["score"]) for row in usable], dtype=float)
            self.model = self._build_model()
            self.model.fit(x, y)
            self.is_available = True
            self.last_error = ""
            self.last_fit_count = len(usable)
            return True
        except Exception as exc:
            self.model = None
            self.is_available = False
            self.last_error = str(exc)
            self.last_fit_count = len(usable)
            return False

    def predict(
        self,
        candidates: list[tuple[float, float, float]],
        *,
        desired_output: float,
        frequency_khz: int,
    ) -> list[float]:
        if not self.is_available or self.model is None or not candidates:
            return [math.nan for _ in candidates]
        x = np.asarray(
            [
                [float(kp), float(ki), float(kd), float(desired_output), float(frequency_khz)]
                for kp, ki, kd in candidates
            ],
            dtype=float,
        )
        try:
            preds = self.model.predict(x)
            return [float(v) for v in preds]
        except Exception as exc:
            self.is_available = False
            self.last_error = str(exc)
            return [math.nan for _ in candidates]


def build_surrogate_training_row(
    kp: float,
    ki: float,
    kd: float,
    *,
    desired_output: float,
    frequency_khz: int,
    score: float,
) -> dict:
    return {
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "desired_output": float(desired_output),
        "frequency_khz": float(frequency_khz),
        "score": float(score),
    }


def build_local_refinement_bounds(
    best_pid: tuple[float, float, float],
    *,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    radius_kp: float,
    radius_ki: float,
    radius_kd: float,
) -> PidBounds:
    """Build a tight refinement box around the current best PID."""
    center = np.asarray(best_pid, dtype=float)
    bounds_max = np.asarray([kp_max, ki_max, kd_max], dtype=float)
    radii = np.asarray(
        [
            max(float(radius_kp), 1e-4),
            max(float(radius_ki), 1e-4),
            max(float(radius_kd), 1e-5),
        ],
        dtype=float,
    )
    lower = np.clip(center - radii, 0.0, bounds_max)
    upper = np.clip(center + radii, 0.0, bounds_max)

    minimum_widths = np.minimum(radii, np.asarray([kp_max, ki_max, kd_max], dtype=float))
    for idx in range(3):
        if upper[idx] > lower[idx]:
            continue
        width = max(float(minimum_widths[idx]), 1e-5)
        if bounds_max[idx] <= 0.0:
            lower[idx] = 0.0
            upper[idx] = 0.0
            continue
        lower[idx] = float(np.clip(center[idx] - width, 0.0, bounds_max[idx]))
        upper[idx] = float(np.clip(center[idx] + width, 0.0, bounds_max[idx]))
        if upper[idx] <= lower[idx]:
            upper[idx] = min(float(bounds_max[idx]), float(lower[idx] + width))
            lower[idx] = max(0.0, float(upper[idx] - width))

    return (
        (float(lower[0]), float(upper[0])),
        (float(lower[1]), float(upper[1])),
        (float(lower[2]), float(upper[2])),
    )


def clamp_pid_to_bounds(pid: tuple[float, float, float], bounds: PidBounds | None) -> tuple[float, float, float]:
    if bounds is None:
        return tuple(float(v) for v in pid)
    values = np.asarray(pid, dtype=float)
    lower = np.asarray([float(pair[0]) for pair in bounds], dtype=float)
    upper = np.asarray([float(pair[1]) for pair in bounds], dtype=float)
    clipped = np.clip(values, lower, upper)
    return float(clipped[0]), float(clipped[1]), float(clipped[2])


def filter_points_to_bounds(
    points: list[tuple[float, float, float]],
    bounds: PidBounds | None,
) -> list[tuple[float, float, float]]:
    if bounds is None:
        return [tuple(float(v) for v in point) for point in points]
    lower = np.asarray([float(pair[0]) for pair in bounds], dtype=float)
    upper = np.asarray([float(pair[1]) for pair in bounds], dtype=float)
    filtered: list[tuple[float, float, float]] = []
    for point in points:
        values = np.asarray(point, dtype=float)
        if np.all(values >= lower) and np.all(values <= upper):
            filtered.append((float(values[0]), float(values[1]), float(values[2])))
    return filtered


def propose_surrogate_candidate(
    surrogate: OnlineSurrogateModel,
    *,
    best_pid: tuple[float, float, float] | None,
    safe_points: list[tuple[float, float, float]],
    observed_points: list[tuple[float, float, float]],
    desired_output: float,
    frequency_khz: int,
    rng: random.Random,
    pool_size: int,
    explore_prob: float,
    jitter_scale: float,
    step_kp: float,
    step_ki: float,
    step_kd: float,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    local_bounds: PidBounds | None = None,
) -> tuple[tuple[float, float, float], float, str]:
    """Generate a bounded candidate pool and score it with the surrogate."""
    global_bounds: PidBounds = (
        (0.0, float(kp_max)),
        (0.0, float(ki_max)),
        (0.0, float(kd_max)),
    )
    active_bounds = local_bounds or global_bounds
    center = clamp_pid_to_bounds(
        best_pid or (0.5 * kp_max, 0.5 * ki_max, 0.5 * kd_max),
        active_bounds,
    )
    lower = np.asarray([float(pair[0]) for pair in active_bounds], dtype=float)
    upper = np.asarray([float(pair[1]) for pair in active_bounds], dtype=float)
    bound_spans = np.maximum(upper - lower, np.asarray([1e-3, 1e-3, 1e-4], dtype=float))
    spans = np.minimum(
        np.asarray([max(step_kp, 1e-3), max(step_ki, 1e-3), max(step_kd, 1e-4)], dtype=float),
        np.maximum(bound_spans * 0.5, np.asarray([1e-3, 1e-3, 1e-4], dtype=float)),
    )
    candidates: list[tuple[float, float, float]] = []
    candidate_safe_points = filter_points_to_bounds(safe_points, active_bounds)

    if best_pid is not None:
        candidates.append(center)

    if candidate_safe_points:
        for point in candidate_safe_points[-min(6, len(candidate_safe_points)) :]:
            candidates.append(tuple(float(v) for v in point))

    for _ in range(max(8, int(pool_size))):
        if best_pid is not None and rng.random() < 0.7:
            base = np.asarray(center, dtype=float)
        elif candidate_safe_points:
            base = np.asarray(rng.choice(candidate_safe_points), dtype=float)
        else:
            base = np.asarray(center, dtype=float)
        noise = np.asarray(
            [
                rng.uniform(-1.0, 1.0) * spans[0] * float(jitter_scale),
                rng.uniform(-1.0, 1.0) * spans[1] * float(jitter_scale),
                rng.uniform(-1.0, 1.0) * spans[2] * float(jitter_scale),
            ],
            dtype=float,
        )
        proposal = np.clip(base + noise, lower, upper)
        candidates.append((float(proposal[0]), float(proposal[1]), float(proposal[2])))

    deduped: list[tuple[float, float, float]] = []
    seen = set()
    observed_rounded = {tuple(round(v, 6) for v in point) for point in observed_points}
    for candidate in candidates:
        rounded = tuple(round(v, 6) for v in candidate)
        if rounded in seen or rounded in observed_rounded:
            continue
        seen.add(rounded)
        deduped.append(candidate)

    if not deduped:
        deduped = [center]

    predictions = surrogate.predict(
        deduped,
        desired_output=desired_output,
        frequency_khz=frequency_khz,
    )
    ranked = sorted(zip(deduped, predictions), key=lambda item: item[1] if np.isfinite(item[1]) else float("inf"))
    if rng.random() < float(explore_prob) and len(ranked) > 1:
        choice = rng.choice(ranked[: min(4, len(ranked))] + ranked[-min(3, len(ranked)) :])
        return choice[0], float(choice[1]), "surrogate_explore"
    best_candidate, best_pred = ranked[0]
    return best_candidate, float(best_pred), "surrogate"


def propose_coordinate_candidate(
    base_pid: tuple[float, float, float],
    axis_index: int,
    axis_direction: float,
    *,
    step_kp: float,
    step_ki: float,
    step_kd: float,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    bounds: PidBounds | None = None,
) -> tuple[tuple[float, float, float], int, float, float]:
    """Adjust exactly one PID term from the current base point."""
    lower_bounds = [0.0, 0.0, 0.0]
    upper_bounds = [float(kp_max), float(ki_max), float(kd_max)]
    if bounds is not None:
        lower_bounds = [float(pair[0]) for pair in bounds]
        upper_bounds = [float(pair[1]) for pair in bounds]
    clamped_base = clamp_pid_to_bounds(base_pid, bounds)
    values = [float(clamped_base[0]), float(clamped_base[1]), float(clamped_base[2])]
    step_sizes = [float(step_kp), float(step_ki), float(step_kd)]

    used_axis = int(axis_index) % 3
    direction = 1.0 if axis_direction >= 0 else -1.0
    delta = direction * step_sizes[used_axis]
    proposed_value = float(np.clip(values[used_axis] + delta, lower_bounds[used_axis], upper_bounds[used_axis]))

    if np.isclose(proposed_value, values[used_axis]):
        direction *= -1.0
        delta = direction * step_sizes[used_axis]
        proposed_value = float(np.clip(values[used_axis] + delta, lower_bounds[used_axis], upper_bounds[used_axis]))

    values[used_axis] = proposed_value
    actual_delta = values[used_axis] - clamped_base[used_axis]
    return (values[0], values[1], values[2]), used_axis, direction, float(actual_delta)


def build_bayes_search_space(
    safe_pids: list[tuple[float, float, float]],
    *,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    pad_kp: float,
    pad_ki: float,
    pad_kd: float,
    best_pid: tuple[float, float, float] | None = None,
    refine_radius_kp: float | None = None,
    refine_radius_ki: float | None = None,
    refine_radius_kd: float | None = None,
) -> list[BayesSpaceDim]:
    """Build a Bayesian search box using local refinement bounds when available."""
    if (
        best_pid is not None
        and refine_radius_kp is not None
        and refine_radius_ki is not None
        and refine_radius_kd is not None
    ):
        bounds = build_local_refinement_bounds(
            best_pid,
            kp_max=kp_max,
            ki_max=ki_max,
            kd_max=kd_max,
            radius_kp=refine_radius_kp,
            radius_ki=refine_radius_ki,
            radius_kd=refine_radius_kd,
        )
        return [
            Real(float(bounds[0][0]), float(bounds[0][1]), name="kp"),
            Real(float(bounds[1][0]), float(bounds[1][1]), name="ki"),
            Real(float(bounds[2][0]), float(bounds[2][1]), name="kd"),
        ]

    if not safe_pids:
        return [
            Real(0.0, kp_max, name="kp"),
            Real(0.0, ki_max, name="ki"),
            Real(0.0, kd_max, name="kd"),
        ]

    mins = np.min(np.asarray(safe_pids, dtype=float), axis=0)
    maxs = np.max(np.asarray(safe_pids, dtype=float), axis=0)
    pads = np.asarray([pad_kp, pad_ki, pad_kd], dtype=float)
    bounds_max = np.asarray([kp_max, ki_max, kd_max], dtype=float)
    lower = np.clip(mins - pads, 0.0, bounds_max)
    upper = np.clip(maxs + pads, 0.0, bounds_max)

    for idx in range(3):
        if upper[idx] <= lower[idx]:
            upper[idx] = min(bounds_max[idx], lower[idx] + max(pads[idx], 1e-3))
            lower[idx] = max(0.0, upper[idx] - max(pads[idx], 1e-3))

    return [
        Real(float(lower[0]), float(upper[0]), name="kp"),
        Real(float(lower[1]), float(upper[1]), name="ki"),
        Real(float(lower[2]), float(upper[2]), name="kd"),
    ]


def filter_seed_points_for_space(
    points: list[tuple[float, float, float]],
    scores: list[float],
    space: list[BayesSpaceDim],
) -> tuple[list[list[float]], list[float]]:
    """Keep only warmup observations that fit inside the Bayesian local box."""
    filtered_points: list[list[float]] = []
    filtered_scores: list[float] = []
    for point, score in zip(points, scores):
        if all(dim.low <= value <= dim.high for dim, value in zip(space, point)):
            filtered_points.append([float(point[0]), float(point[1]), float(point[2])])
            filtered_scores.append(float(score))
    return filtered_points, filtered_scores

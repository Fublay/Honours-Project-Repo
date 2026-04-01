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
) -> tuple[tuple[float, float, float], float, str]:
    """Generate a bounded candidate pool and score it with the surrogate."""
    center = best_pid or (0.5 * kp_max, 0.5 * ki_max, 0.5 * kd_max)
    spans = np.asarray([max(step_kp, 1e-3), max(step_ki, 1e-3), max(step_kd, 1e-3)], dtype=float)
    candidates: list[tuple[float, float, float]] = []

    if safe_points:
        for point in safe_points[-min(6, len(safe_points)) :]:
            candidates.append(tuple(float(v) for v in point))

    for _ in range(max(8, int(pool_size))):
        if rng.random() < 0.5 and best_pid is not None:
            base = np.asarray(center, dtype=float)
        elif safe_points:
            base = np.asarray(rng.choice(safe_points), dtype=float)
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
        proposal = np.clip(base + noise, [0.0, 0.0, 0.0], [kp_max, ki_max, kd_max])
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
        deduped = [tuple(float(v) for v in center)]

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
) -> tuple[tuple[float, float, float], int, float, float]:
    """Adjust exactly one PID term from the current base point."""
    values = [float(base_pid[0]), float(base_pid[1]), float(base_pid[2])]
    step_sizes = [float(step_kp), float(step_ki), float(step_kd)]
    max_values = [float(kp_max), float(ki_max), float(kd_max)]

    used_axis = int(axis_index) % 3
    direction = 1.0 if axis_direction >= 0 else -1.0
    delta = direction * step_sizes[used_axis]
    proposed_value = float(np.clip(values[used_axis] + delta, 0.0, max_values[used_axis]))

    if np.isclose(proposed_value, values[used_axis]):
        direction *= -1.0
        delta = direction * step_sizes[used_axis]
        proposed_value = float(np.clip(values[used_axis] + delta, 0.0, max_values[used_axis]))

    values[used_axis] = proposed_value
    actual_delta = values[used_axis] - base_pid[used_axis]
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
) -> list[BayesSpaceDim]:
    """Build a local Bayesian search box around the safe region from bootstrap."""
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

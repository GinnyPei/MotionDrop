"""Quadratic latency model and SLO-to-token-budget inversion."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Default response-latency target for SLO-driven budgets.
DEFAULT_SLO_MS = 150.0


@dataclass(frozen=True)
class BudgetDecision:
    retained_tokens: int
    predicted_latency_ms: float | None
    slo_ms: float | None
    mode: str


@dataclass(frozen=True)
class LatencyModel:
    """Paper Equation (7), expressed in milliseconds."""

    alpha1: float
    alpha2: float
    beta1: float
    beta2: float
    gamma: float
    metadata: dict | None = None

    def __post_init__(self) -> None:
        for name in ("alpha1", "alpha2", "beta1", "beta2", "gamma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def encoder_ms(self, visual_tokens: int) -> float:
        value = max(0, int(visual_tokens))
        return self.alpha1 * value + self.alpha2 * value**2

    def prefill_ms(self, retained_tokens: int) -> float:
        value = max(0, int(retained_tokens))
        return self.beta1 * value + self.beta2 * value**2

    def total_ms(self, visual_tokens: int, retained_tokens: int) -> float:
        return self.encoder_ms(visual_tokens) + self.prefill_ms(retained_tokens) + self.gamma

    def largest_feasible_budget(
        self,
        *,
        visual_tokens: int,
        slo_ms: float,
        minimum: int,
        maximum: int,
    ) -> BudgetDecision:
        if slo_ms <= 0.0:
            raise ValueError("slo_ms must be positive")
        if minimum < 1 or maximum < minimum:
            raise ValueError("invalid token-budget bounds")
        minimum_latency = self.total_ms(visual_tokens, minimum)
        if minimum_latency > slo_ms:
            raise ValueError(
                "The fitted latency model cannot satisfy the SLO even at the minimum "
                f"budget: predicted={minimum_latency:.3f} ms, slo={slo_ms:.3f} ms"
            )
        low, high = minimum, maximum
        while low < high:
            middle = (low + high + 1) // 2
            if self.total_ms(visual_tokens, middle) <= slo_ms:
                low = middle
            else:
                high = middle - 1
        return BudgetDecision(
            retained_tokens=low,
            predicted_latency_ms=self.total_ms(visual_tokens, low),
            slo_ms=float(slo_ms),
            mode="slo_fitted_quadratic",
        )

    def to_dict(self) -> dict:
        return {
            "schema": "motiondrop_latency_model_v1",
            "units": "milliseconds",
            "equation": (
                "alpha1*visual_tokens + alpha2*visual_tokens^2 + "
                "beta1*retained_tokens + beta2*retained_tokens^2 + gamma"
            ),
            "coefficients": {
                key: value for key, value in asdict(self).items() if key != "metadata"
            },
            "metadata": self.metadata or {},
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict) -> LatencyModel:
        if data.get("schema") != "motiondrop_latency_model_v1":
            raise ValueError("unsupported latency-model schema")
        coefficients = data["coefficients"]
        return cls(
            alpha1=float(coefficients["alpha1"]),
            alpha2=float(coefficients["alpha2"]),
            beta1=float(coefficients["beta1"]),
            beta2=float(coefficients["beta2"]),
            gamma=float(coefficients["gamma"]),
            metadata=data.get("metadata") or {},
        )

    @classmethod
    def load(cls, path: str | Path) -> LatencyModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fixed_budget(
    *,
    original_tokens: int,
    temporal_units: int,
    compression: float,
) -> BudgetDecision:
    if not 0.0 <= compression < 1.0:
        raise ValueError("compression must be in [0, 1)")
    retained = max(temporal_units, round(original_tokens * (1.0 - compression)))
    retained = min(original_tokens, retained)
    return BudgetDecision(retained, None, None, "fixed_compression")


def _nonnegative_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve this five-variable NNLS problem by enumerating active sets."""

    columns = design.shape[1]
    best_coefficients = np.zeros(columns, dtype=np.float64)
    best_error = float(np.square(target).sum())
    for count in range(1, columns + 1):
        for active in itertools.combinations(range(columns), count):
            values, *_ = np.linalg.lstsq(design[:, active], target, rcond=None)
            if np.any(values < 0.0):
                continue
            coefficients = np.zeros(columns, dtype=np.float64)
            coefficients[list(active)] = values
            error = float(np.square(design @ coefficients - target).sum())
            if error < best_error:
                best_error = error
                best_coefficients = coefficients
    return best_coefficients


def fit_latency_model(
    encoder_samples: Iterable[dict],
    prefill_samples: Iterable[dict],
    *,
    metadata: dict | None = None,
) -> tuple[LatencyModel, dict]:
    """Fit Equations (6)-(7) from separately measured encoder/prefill samples."""

    encoder = [
        (int(sample["visual_tokens"]), float(sample["latency_ms"])) for sample in encoder_samples
    ]
    prefill = [
        (int(sample["retained_tokens"]), float(sample["latency_ms"])) for sample in prefill_samples
    ]
    if len({value for value, _ in encoder}) < 3:
        raise ValueError("at least three distinct encoder token counts are required")
    if len({value for value, _ in prefill}) < 3:
        raise ValueError("at least three distinct prefill token counts are required")
    if any(tokens < 1 or latency <= 0.0 for tokens, latency in encoder + prefill):
        raise ValueError("profile token counts and latencies must be positive")

    rows: list[list[float]] = []
    targets: list[float] = []
    for visual_tokens, encoder_ms in encoder:
        for retained_tokens, prefill_ms in prefill:
            rows.append(
                [
                    visual_tokens,
                    visual_tokens**2,
                    retained_tokens,
                    retained_tokens**2,
                    1.0,
                ]
            )
            targets.append(encoder_ms + prefill_ms)
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    scale = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    normalized = design / scale
    coefficients = _nonnegative_least_squares(normalized, target) / scale
    prediction = design @ coefficients
    residual = target - prediction
    denominator = float(np.square(target - target.mean()).sum())
    metrics = {
        "combined_samples": int(target.size),
        "rmse_ms": float(np.sqrt(np.square(residual).mean())),
        "max_absolute_error_ms": float(np.abs(residual).max()),
        "r_squared": 1.0
        if denominator == 0.0
        else 1.0 - float(np.square(residual).sum()) / denominator,
    }
    model = LatencyModel(*[float(value) for value in coefficients], metadata=metadata)
    return model, metrics


__all__ = [
    "DEFAULT_SLO_MS",
    "BudgetDecision",
    "LatencyModel",
    "fit_latency_model",
    "fixed_budget",
]

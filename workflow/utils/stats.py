"""Calculate summary statistics"""

from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau, pearsonr

METRIC_FNS: dict[str, "MetricFnProto"] = {}


class MetricFnProto(Protocol):
    def __call__(self, y_pred: npt.NDArray, y_ref: npt.NDArray) -> float: ...

    __name__: str


def register_metric_fn(func: MetricFnProto) -> MetricFnProto:
    """Register a metric function."""

    metric_name = func.__name__.split("get_")[1]
    METRIC_FNS[metric_name] = func
    return func


@register_metric_fn
def get_mue(y_pred: npt.NDArray, y_ref: npt.NDArray) -> float:
    """Calculate the mean unsigned error (MUE) between predicted and reference values."""
    return float(np.mean(np.abs(y_pred - y_ref)))


@register_metric_fn
def get_rmse(y_pred: npt.NDArray, y_ref: npt.NDArray) -> float:
    """Calculate the root mean squared error (RMSE) between predicted and reference values."""
    return float(np.sqrt(np.mean((y_pred - y_ref) ** 2)))


@register_metric_fn
def get_kendall_tau(y_pred: npt.NDArray, y_ref: npt.NDArray) -> float:
    """Calculate the Kendall Tau correlation coefficient between predicted and reference values."""
    return float(kendalltau(y_pred, y_ref).correlation)


@register_metric_fn
def get_r_squared(y_pred: npt.NDArray, y_ref: npt.NDArray) -> float:
    """Calculate the R-squared value between predicted and reference values."""
    return float(pearsonr(y_pred, y_ref)[0]) ** 2


def get_bootstrapped_data(
    y_pred: np.ndarray, y_ref: np.ndarray, n_samples: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    """Generate bootstrapped samples of predicted and reference values."""
    indices = np.random.randint(0, len(y_pred), size=(n_samples, len(y_pred)))
    y_pred_bootstrap = y_pred[indices]
    y_ref_bootstrap = y_ref[indices]
    return y_pred_bootstrap, y_ref_bootstrap


def get_bootstrapped_metric(
    metric_fn: MetricFnProto,
    y_pred: np.ndarray,
    y_ref: np.ndarray,
    n_samples: int = 10_000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate a bootstrapped metric with confidence intervals."""
    y_pred_bootstrap, y_ref_bootstrap = get_bootstrapped_data(y_pred, y_ref, n_samples)
    metrics = np.array(
        [metric_fn(pred, ref) for pred, ref in zip(y_pred_bootstrap, y_ref_bootstrap)]
    )

    mean_metric = float(np.mean(metrics))
    lower_bound = float(np.percentile(metrics, (1 - ci) / 2 * 100))
    upper_bound = float(np.percentile(metrics, (1 + ci) / 2 * 100))

    return mean_metric, lower_bound, upper_bound

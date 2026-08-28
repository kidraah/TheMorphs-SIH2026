"""nowcast_eval -- a ruler for severe-weather nowcasts.

It measures; it does not predict. Every number it returns is a
deterministic function of (predictions, observations, config).
"""
from .config import EvalConfig
from .contingency import ContingencyTable, contingency
from .core import EvaluationResult, evaluate, evaluate_predict_fn
from .fss import fss, useful_scale_threshold
from .bootstrap import (CIResult, attach_confidence_intervals, bootstrap_ci,
                        significantly_better)
from .multihazard import MultiHazardResult, evaluate_multi
from .probabilistic import brier_score, probabilistic_scores, reliability_curve

__all__ = [
    "EvalConfig", "ContingencyTable", "contingency",
    "EvaluationResult", "evaluate", "evaluate_predict_fn",
    "fss", "useful_scale_threshold",
    "brier_score", "probabilistic_scores", "reliability_curve",
    "CIResult", "bootstrap_ci", "attach_confidence_intervals", "significantly_better",
    "MultiHazardResult", "evaluate_multi",
]
__version__ = "0.1.0"

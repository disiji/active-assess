import numpy as np
import warnings
warnings.filterwarnings("ignore")
from typing import List

NUM_BINS = 20

def eval_ece(confidences: List[float], observations: List[bool], num_bins=NUM_BINS):
    """
    Evaluate ECE given a list of samples with equal-width binning.
    :param confidences: List[float]
        A list of prediction scores.
    :param observations: List[bool]
        A list of boolean observations.
    :param num_bins: int
        The number of bins used to estimate ECE. Default: 10
    :return: float
    """
    confidences = np.array(confidences)
    observations = np.array(observations) * 1.0
    bins = np.linspace(0, 1, num_bins + 1)
    digitized = np.digitize(confidences, bins[1:-1])

    w = np.array([(digitized == i).sum() for i in range(num_bins)])
    w = w / sum(w)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        confidence_bins = np.array([confidences[digitized == i].mean() for i in range(num_bins)])
        accuracy_bins = np.array([observations[digitized == i].mean() for i in range(num_bins)])
    confidence_bins[np.isnan(confidence_bins)] = 0
    accuracy_bins[np.isnan(accuracy_bins)] = 0
    diff = np.absolute(confidence_bins - accuracy_bins)
    ece = np.inner(diff, w)
    return ece


def mean_reciprocal_rank(metric_val: np.ndarray,
                         ground_truth: np.ndarray,
                         mode: str) -> float:
    """Computes mean reciprocal rank"""
    num_classes = metric_val.shape[0]
    k = np.sum(ground_truth)

    # Compute rank of each class
    argsort = metric_val.argsort()
    rank = np.empty_like(argsort)
    rank[argsort] = np.arange(num_classes) + 1
    if mode == 'max':  # Need to flip so that largest class has rank 1
        rank = num_classes - rank + 1

    # In top-k setting, we need to adjust so that other ground truth classes
    # are not considered in the ranking.
    raw_rank = rank[ground_truth]
    argsort = raw_rank.argsort()
    offset = np.empty_like(argsort)
    offset[argsort] = np.arange(k)
    adjusted_rank = raw_rank - offset

    return (1 / adjusted_rank).mean()
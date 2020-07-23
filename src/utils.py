import argparse
import ctypes
import random
from collections import deque
from functools import reduce
from multiprocessing import Array

import matplotlib.pyplot as plt

from data_utils import *
from models import BetaBernoulli
from sampling import SAMPLE_CATEGORY

COLUMN_WIDTH = 3.25  # Inches
GOLDEN_RATIO = 1.61803398875
FONT_SIZE = 8

RUNS = 100
LOG_FREQ = 100
PRIOR_STRENGTH = 3
HOLDOUT_RATIO = 0.1


#########################METRIC##########################
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
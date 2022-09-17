from typing import List, Optional, Callable, Union, Any, Tuple

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

from .metrics import concordance_index_censored as ci


def concordance_index(
    y_true: Union[Tensor, ndarray], 
    y_pred: Union[Tensor, ndarray]
) -> float:
    """Compute the concordance-index value.

    For coxph model
    Args:
        y_true (Union[Tensor, ndarray]): Observed time. Negative values are considered right censored.
        y_pred (Union[Tensor, ndarray]): Predicted value (proportional hazard).

    For discrete model
    Args:
        y_true (Union[Tensor, ndarray]): Observed time (at the first column) and censorship (at the second column). 
        y_pred (Union[Tensor, ndarray]): Predicted value (time-dependent hazard function).
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()

    if y_true.shape[1] == 1: # coxph model
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        t = np.abs(y_true)
        e = (y_true > 0).astype(np.int32)
        return ci(e, t, -y_pred, tied_tol=1e-08)[0]
    else: # discrete model
        y_t, y_e = y_true[:, 0], (1 - y_true[:, 1]).astype(np.bool_)
        survival = np.cumprod(1.0 - y_pred, axis=1)
        risk = np.sum(survival, axis=1)
        return ci(y_e, y_t, -risk, tied_tol=1e-08)[0]


def evaluator(y, y_hat, metrics='cindex', **kws):
    """
    If it is a discrete model:
        y: [B, 2] (col1: y_t, col2: y_c)
        y_hat: [B, BINS]
    else:
        y: [B, 1]
        y_hat: [B, 1]
    """
    assert metrics in ['cindex', 'auc']

    if metrics == 'cindex':
        return concordance_index(y, y_hat)
    else:
        raise NotImplementedError(f"Metrics {metrics} has not implemented.")

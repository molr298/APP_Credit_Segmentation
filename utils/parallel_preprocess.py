import numpy as np
import pandas as pd
from numba import njit, prange
from pytest import approx
from sklearn.preprocessing import StandardScaler


@njit(parallel=True)
def standard_parallel(A):
    """
    Standardise data by removing the mean and scaling to unit variance,
    equivalent to sklearn StandardScaler.
    
    Uses explicit parallel loop; may offer improved performance in some
    cases.
    """
    n = A.shape[1]
    res = np.empty_like(A, dtype=np.float64)

    for i in prange(n):
        data_i = A[:, i]
        res[:, i] = (data_i - np.mean(data_i)) / np.std(data_i)

    return res
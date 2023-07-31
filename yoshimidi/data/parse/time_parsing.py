import math

import numpy as np
import torch

# See notebooks/time_distribution.ipynb for the derivation of these values.
_TIME_SUPPORTS = [*np.arange(0, 162 + 1, 162 / 6), 255.0]
NUM_TIME_SUPPORTS = len(_TIME_SUPPORTS)  # 8
_TIME_LOG_MIN = -7.6246189861593985
_TIME_LOG_MAX = 5.758507374345448


def time_to_uint8(time: float) -> int:
    if time <= 0:
        return 0
    time_log = math.log(time)
    time_norm = (time_log - _TIME_LOG_MIN) / (_TIME_LOG_MAX - _TIME_LOG_MIN)
    return int(time_norm * 2**8)


def time_from_uint8(time_uint8: int) -> float:
    if time_uint8 == 0:
        return 0.0
    time_norm = float(time_uint8) / 2**8
    time_log = time_norm * (_TIME_LOG_MAX - _TIME_LOG_MIN) + _TIME_LOG_MIN
    time = math.exp(time_log)
    assert time > -1e-3
    time = max(time, 0)
    return time


def time_uint8_to_support(time: int, output: torch.Tensor) -> None:
    assert time >= 0
    assert output.shape == (len(_TIME_SUPPORTS),)
    if time == 0:
        output[0] = 1
        return
    if time >= _TIME_SUPPORTS[-1]:
        output[-1] = 1
        return
    upper_idx = next(i for i, support in enumerate(_TIME_SUPPORTS) if support > time)
    lower_idx = upper_idx - 1
    lower_support = _TIME_SUPPORTS[lower_idx]
    upper_support = _TIME_SUPPORTS[upper_idx]
    assert lower_support <= time < upper_support, (lower_support, time, upper_support)
    weighting = (time - lower_support) / (upper_support - lower_support)
    output[lower_idx] = 1 - weighting
    output[upper_idx] = weighting


def time_uint8_from_support(support: np.ndarray) -> int:
    assert support.shape == (len(_TIME_SUPPORTS),)
    return np.dot(support, _TIME_SUPPORTS)

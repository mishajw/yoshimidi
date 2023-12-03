import math

import numpy as np
import torch

# See notebooks/time_distribution.ipynb for the derivation of these values.
_TIME_SUPPORTS = [*np.arange(0, 162 + 1, 162 / 14), 255.0]
NUM_TIME_SUPPORTS = len(_TIME_SUPPORTS)  # 16
_TIME_LOG_MIN = -7.6246189861593985
_TIME_LOG_MAX = 5.758507374345448
_time_supports_torch = torch.Tensor(_TIME_SUPPORTS)


def time_to_uint8(time: float) -> int:
    if time <= 0:
        return 0
    time_log = math.log(time)
    time_norm = (time_log - _TIME_LOG_MIN) / (_TIME_LOG_MAX - _TIME_LOG_MIN)
    time_norm = max(time_norm, 0)
    time_norm = min(time_norm, 1)
    time_uint8 = int(time_norm * 2**8)
    if time_uint8 == 256:
        time_uint8 = 255
    assert time_uint8 >= 0 and time_uint8 <= 255, time_uint8
    return time_uint8


def time_from_uint8(time_uint8: int | float) -> float:
    if time_uint8 == 0:
        return 0.0
    time_norm = float(time_uint8) / 2**8
    time_log = time_norm * (_TIME_LOG_MAX - _TIME_LOG_MIN) + _TIME_LOG_MIN
    time = math.exp(time_log)
    assert time > -1e-3
    time = max(time, 0)
    return time


def time_uint8_to_support(time: torch.Tensor, output: torch.Tensor) -> None:
    global _time_supports_torch
    if _time_supports_torch.device != time.device:
        _time_supports_torch = _time_supports_torch.to(time.device)
    assert (time >= 0).all() and (time <= 255).all()
    assert output.shape == (time.shape[0], len(_TIME_SUPPORTS)), (
        output.shape,
        time.shape[0],
        len(_TIME_SUPPORTS),
    )
    time = time.clamp(max=254)  # clamp to avoid out-of-bounds indexing
    upper_idx = torch.searchsorted(_time_supports_torch, time, side="right")
    lower_idx = upper_idx - 1
    lower_support = torch.index_select(_time_supports_torch, 0, lower_idx)
    upper_support = torch.index_select(_time_supports_torch, 0, upper_idx)
    assert torch.all((lower_support <= time) & (time <= upper_support)), (
        lower_support,
        time,
        upper_support,
    )
    weighting = (time - lower_support) / (upper_support - lower_support)
    output[torch.arange(time.size(0)), lower_idx] = 1 - weighting
    output[torch.arange(time.size(0)), upper_idx] = weighting


def time_uint8_from_support(support: torch.Tensor) -> float:
    assert support.shape == (len(_TIME_SUPPORTS),), support.shape
    return torch.dot(
        support,
        torch.tensor(_TIME_SUPPORTS, dtype=support.dtype, device=support.device),
    ).item()

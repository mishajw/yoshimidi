import math

import pytest
import torch

from yoshimidi.data.parse import time_parsing


@pytest.mark.parametrize("time", [0.0, 0.5, 1.0, 1.5, 2.0, 100.0])
def test_uint8_conversion(time: float) -> None:
    assert math.isclose(
        time,
        time_parsing.time_from_uint8(time_parsing.time_to_uint8(time)),
        # Lots of allowance for error, as we're converting from a uint8 to a float.
        rel_tol=0.05,
    )


@pytest.mark.parametrize("time_uint8", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 255])
def test_support_conversion(time_uint8: int) -> None:
    supports = torch.zeros(time_parsing.NUM_TIME_SUPPORTS, dtype=torch.float32)
    time_parsing.time_uint8_to_support(time_uint8, supports)
    time_uint8_recons = time_parsing.time_uint8_from_support(supports)
    assert time_uint8 == time_uint8_recons

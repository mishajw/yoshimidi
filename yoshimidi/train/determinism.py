import contextlib
import hashlib
import random
from typing import Any

import numpy as np
import torch

# TODO: What would be great is if we could enforce that this function is always called
# for all "nondeterministic" code. For example, we could somehow inject code into the
# RNGs used st. it fails if this hasn't been called first.
#
# However, we'd still need to guarantee that the tags are set properly: For example, if
# you want to call `determinism` in a loop, you'd have to use a new tag every time.
# While we could keep track of the tags, this would probably be inefficient (unless you
# could do this at compile time - rewrite in Rust & use the borrow checker? :)).
#
# Also, god knows how possible it is to inject code into the CUDA RNGs. Obviously
# wouldn't be possible on the kernel-level, but perhaps at the Python API level?


@contextlib.contextmanager
def determinism(tag: str) -> Any:
    """Runs code within the context manager with a fixed seed.

    Ideally, we'd use this everywhere we have a nondeterministic operation, and always
    call with a different tag. This way we can guarantee code is deterministic without
    having to save out RNG states.
    """

    seed = int(hashlib.sha256(tag.encode()).hexdigest(), 16) % 2**32
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()
    torch_cuda_random_state = (
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        torch.set_rng_state(torch_random_state)
        if torch.cuda.is_available() and torch_cuda_random_state is not None:
            torch.cuda.set_rng_state(torch_cuda_random_state)
        torch.use_deterministic_algorithms(False)

"""Shared pytest setup.

nanochat.common runs its dtype autodetect at import time, which queries the CUDA
device capability. On WSL2 + Pascal this leaves a stale cudaErrorNotSupported on
the context, and the next tracked CUDA call raises it. compute_init() flushes
that for the training/eval scripts; mirror it here so tests that do their own
CUDA work without calling compute_init don't trip over it.
"""
import pytest
import torch


@pytest.fixture(autouse=True)
def _flush_stale_cuda_error():
    if torch.cuda.is_available():
        torch.cuda.init()
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass  # raising it clears the pending error
    yield

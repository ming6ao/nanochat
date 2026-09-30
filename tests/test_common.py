"""Tests for the peak-FLOPS lookup in nanochat.common."""
from nanochat.common import get_peak_flops


def test_peak_flops_known_modern_gpu():
    assert get_peak_flops("NVIDIA H100 SXM") == 989e12
    assert get_peak_flops("NVIDIA A100-SXM4-80GB") == 312e12


def test_peak_flops_gtx_1080_ti():
    # Pascal has no tensor cores; the table stores the FP32 CUDA-core peak.
    assert get_peak_flops("NVIDIA GeForce GTX 1080 Ti") == 11.34e12


def test_peak_flops_unknown_device_is_inf():
    # Unchanged legacy behavior: unknown GPUs return infinity.
    assert get_peak_flops("Some Made Up GPU XYZ") == float("inf")

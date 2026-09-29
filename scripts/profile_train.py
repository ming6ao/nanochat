"""Profile a nanochat ``scripts.base_train`` run and save PyTorch profiler artifacts.

This is a thin wrapper: it starts a ``torch.profiler`` session, runs the real
trainer in-process via ``runpy``, then exports the trace, a per-operator table,
a small summary, and a Markdown report.

Usage
-----
    python -m scripts.profile_train --steps 100 --out profiles/d8_s512 -- \
        --depth=8 --window-pattern=L --max-seq-len=512 \
        --device-batch-size=8 --total-batch-size=4096 \
        --warmup-steps=3 --eval-every=-1 --core-metric-every=-1 \
        --sample-every=-1 --save-every=-1 --run=dummy --model-tag=prof_d8

Everything after ``--`` is forwarded verbatim to ``scripts.base_train``. If
``--num-iterations`` is not supplied to the trainer, it is set to ``--steps``.

Artifacts written to ``--out`` (default ``profiles/<tag>``):
  summary.json, key_averages.json, train.log, chrome_trace.json,
  tensorboard/, PROFILE_REPORT.md

Notes
-----
* On WSL2, CUPTI cannot initialize, so CUDA kernel activities are unavailable and
  only CPU/operator + memory events are captured. The identical command on native
  Linux will also capture CUDA kernel timings automatically.
* ``--cpu-only`` forces a CPU-only profile even where CUDA activities work.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import runpy
import sys
import time


class _Tee(io.TextIOBase):
    """Write to several streams at once (real stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            if st is not None:
                st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            if st is not None:
                st.flush()


def _split_args(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _parse_peak_memory(log_path):
    if not os.path.exists(log_path):
        return None
    pat = re.compile(r"Peak memory usage:\s*([\d.]+)\s*MiB")
    peak = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                peak = float(m.group(1))
    return peak


def main():
    prof_argv, train_argv = _split_args(sys.argv[1:])

    ap = argparse.ArgumentParser(description="Profile scripts.base_train and save artifacts")
    ap.add_argument("--steps", type=int, default=100, help="optimizer steps to run (sets --num-iterations)")
    ap.add_argument("--out", default=None, help="output directory (default: profiles/<tag>)")
    ap.add_argument("--tag", default="profile", help="profile tag used when --out is not given")
    ap.add_argument("--no-memory", action="store_true", help="disable profiler memory recording")
    ap.add_argument("--no-shapes", action="store_true", help="disable profiler shape recording")
    ap.add_argument("--cpu-only", action="store_true", help="do not request CUDA profiler activities")
    args = ap.parse_args(prof_argv)

    out_dir = args.out or os.path.join("profiles", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")

    # Ensure the trainer runs exactly --steps optimizer steps.
    if not any(a == "--num-iterations" or a.startswith("--num-iterations=") for a in train_argv):
        train_argv = [f"--num-iterations={args.steps}", *train_argv]

    import torch
    from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

    is_wsl = "microsoft" in platform.uname().release.lower()
    cuda_available = torch.cuda.is_available()

    activities = [ProfilerActivity.CPU]
    want_cuda = cuda_available and not args.cpu_only
    if want_cuda:
        activities.append(ProfilerActivity.CUDA)

    # Run the trainer with our argv, capturing stdout to both the terminal and a log.
    sys.argv = ["scripts/base_train.py", *train_argv]
    log_file = open(log_path, "w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_file)

    print(f"[profile_train] out_dir      : {out_dir}")
    print(f"[profile_train] activities   : {[a.name for a in activities]}")
    print(f"[profile_train] wsl2         : {is_wsl}  (CUDA kernel activities: {want_cuda and not is_wsl})")
    print(f"[profile_train] trainer args : {' '.join(train_argv)}")
    sys.stdout.flush()

    t0 = time.time()
    profiler = profile(
        activities=activities,
        record_shapes=not args.no_shapes,
        profile_memory=not args.no_memory,
        on_trace_ready=tensorboard_trace_handler(tb_dir, use_gzip=True),
    )
    with profiler as prof:
        with contextlib.redirect_stdout(tee):
            try:
                runpy.run_module("scripts.base_train", run_name="__main__", alter_sys=False)
            except SystemExit:
                pass
    wall_time = time.time() - t0
    log_file.flush()

    # Restore GC (the trainer disables it during the timed loop).
    import gc

    gc.enable()

    # The trace handler wrote <worker>.<ts>.pt.trace.json.gz at profiler exit; copy it
    # to a stable, easy-to-find name. (Calling export_chrome_trace again here would
    # raise "Trace is already saved".)
    import glob
    import shutil

    chrome_path = os.path.join(out_dir, "chrome_trace.json.gz")
    candidates = sorted(glob.glob(os.path.join(tb_dir, "**", "*.pt.trace.json.gz"), recursive=True))
    if candidates:
        shutil.copyfile(candidates[-1], chrome_path)

    # Export the aggregated per-operator table as JSON.
    rows = []
    for e in prof.key_averages():
        rows.append(
            {
                "key": e.key,
                "count": e.count,
                "self_cpu_time_total_us": e.self_cpu_time_total,
                "cpu_time_total_us": e.cpu_time_total,
                "self_device_time_total_us": e.self_device_time_total,
                "device_time_total_us": e.device_time_total,
                "self_cpu_memory_usage": e.self_cpu_memory_usage,
                "cpu_memory_usage": e.cpu_memory_usage,
                "self_device_memory_usage": e.self_device_memory_usage,
                "device_memory_usage": e.device_memory_usage,
                "input_shapes": e.input_shapes,
            }
        )
    ka_path = os.path.join(out_dir, "key_averages.json")
    with open(ka_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    has_dev_time = any(r["self_device_time_total_us"] > 0 for r in rows)
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "profile_dir": os.path.abspath(out_dir),
        "steps": args.steps,
        "train_args": train_argv,
        "wall_time_s": wall_time,
        "peak_memory_mib": _parse_peak_memory(log_path),
        "cuda_available": cuda_available,
        "is_wsl": is_wsl,
        "activities": [a.name for a in activities],
        "cuda_activities_captured": has_dev_time,
        "num_operator_keys": len(rows),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Build the Markdown report.
    sys.path.insert(0, os.getcwd())
    from scripts.profile_report import generate_report

    report_path = generate_report(out_dir)

    print()
    print("[profile_train] done")
    print(f"[profile_train] wall time    : {wall_time:.1f}s")
    print(f"[profile_train] chrome trace : {chrome_path}")
    print(f"[profile_train] key averages : {ka_path}")
    print(f"[profile_train] report       : {report_path}")
    print(f"[profile_train] tensorboard  : {tb_dir}/")
    if is_wsl and want_cuda:
        print(
            "[profile_train] note: WSL2 detected -> CUPTI/kernel activities are unavailable; "
            "CPU/operator + memory events were captured instead."
        )


if __name__ == "__main__":
    main()

"""Generate a Markdown report from a nanochat PyTorch profiler run.

The profile directory is produced by ``scripts/profile_train.py`` and contains:

  * ``key_averages.json``  -- per-operator aggregated timings / memory
  * ``summary.json``       -- run metadata (config, wall time, peak memory, ...)
  * ``train.log``          -- captured stdout of ``scripts.base_train``
  * ``chrome_trace.json``  -- Chrome/Perfetto trace (open at chrome://tracing or perfetto.dev)
  * ``tensorboard/``       -- TensorBoard trace (``*.pt.trace.json.gz``)

Usage:
    python -m scripts.profile_report --dir profiles/d8_s512
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from datetime import datetime

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+).*?loss:\s*([-\d.eE+]+).*?dt:\s*([\d.]+)ms.*?tok/sec:\s*([\d,]+)"
)


def _us(x: float) -> str:
    x = float(x or 0.0)
    if x >= 1e6:
        return f"{x / 1e6:.2f} s"
    if x >= 1e3:
        return f"{x / 1e3:.2f} ms"
    return f"{x:.1f} us"


def _bytes(x: float) -> str:
    x = float(x or 0.0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} PiB"


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def _parse_steps(log_path: str):
    steps = []
    if not os.path.exists(log_path):
        return steps
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(
                    {
                        "step": int(m.group(1)),
                        "total": int(m.group(2)),
                        "loss": float(m.group(3)),
                        "dt_ms": float(m.group(4)),
                        "tok_per_s": int(m.group(5).replace(",", "")),
                    }
                )
    return steps


def _categorize(key: str) -> str:
    k = key.lower()
    if any(t in k for t in ("local_scalar_dense", "::item", "tolist")):
        return "device sync"
    if any(t in k for t in ("scaled_dot_product", "attention", "softmax", "bmm", "_flash")):
        return "attention"
    if any(t in k for t in ("addmm", "matmul", "::mm", "linear", "einsum")):
        return "matmul/linear"
    if any(t in k for t in ("optim", "muon", "zerograd", "foreach")):
        return "optimizer"
    if any(t in k for t in ("silu", "gelu", "rms_norm", "rmsnorm", "layer_norm", "softcap")):
        return "norms/activations"
    if any(t in k for t in ("embedding", "index", "gather", "scatter")):
        return "embedding/gather"
    if any(t in k for t in ("copy_", "to_", "::to", "clone", "contiguous", "cat")):
        return "copies/layout"
    if any(t in k for t in ("mul", "add", "div", "sub", "pow", "neg", "clamp")):
        return "elementwise"
    return "other"


def _summary_stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals = sorted(vals)
    return {
        "n": len(vals),
        "min": vals[0],
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "max": vals[-1],
    }


def generate_report(profile_dir: str, out_path: str | None = None) -> str:
    ka_path = os.path.join(profile_dir, "key_averages.json")
    if not os.path.exists(ka_path):
        raise FileNotFoundError(f"{ka_path} not found; run scripts/profile_train.py first")

    with open(ka_path) as f:
        rows = json.load(f)
    summary = {}
    sp = os.path.join(profile_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            summary = json.load(f)
    steps = _parse_steps(os.path.join(profile_dir, "train.log"))

    total_cpu = sum(r.get("self_cpu_time_total_us", 0.0) for r in rows)
    total_dev = sum(r.get("self_device_time_total_us", 0.0) for r in rows)
    has_dev_time = total_dev > 0
    total_dev_mem = sum(r.get("self_device_memory_usage", 0) for r in rows)
    n_calls = sum(r.get("count", 0) for r in rows)

    by_cpu = sorted(rows, key=lambda r: r.get("self_cpu_time_total_us", 0.0), reverse=True)
    by_dev = sorted(rows, key=lambda r: r.get("self_device_time_total_us", 0.0), reverse=True)
    by_mem = sorted(rows, key=lambda r: r.get("self_device_memory_usage", 0), reverse=True)

    # category breakdown by CPU time
    cats: dict[str, float] = {}
    for r in rows:
        cats[_categorize(r.get("key", ""))] = cats.get(_categorize(r.get("key", "")), 0.0) + r.get(
            "self_cpu_time_total_us", 0.0
        )

    cfg = " ".join(summary.get("train_args", [])) or "(unknown)"
    wall = summary.get("wall_time_s")
    peak = summary.get("peak_memory_mib")
    activities = summary.get("activities", [])
    cuda_available = summary.get("cuda_available")
    wsl = summary.get("is_wsl")

    lines: list[str] = []
    lines.append("# PyTorch Training Profile Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- **Profile directory:** `{profile_dir}`")
    lines.append(f"- **Trainer config:** `{cfg}`")
    lines.append(f"- **Profiler activities:** {', '.join(activities) if activities else '(unknown)'}")
    lines.append(f"- **CUDA available:** {cuda_available}   (WSL2: {wsl})")
    lines.append(f"- **CUDA/kernel activities captured:** {has_dev_time}")
    if wall is not None:
        lines.append(f"- **Profiled wall time:** {wall:.1f} s")
    if peak is not None:
        lines.append(f"- **Peak CUDA memory (from trainer):** {peak:.0f} MiB")
    lines.append(f"- **Aggregated operator events:** {n_calls:,} across {len(rows):,} distinct ops")
    lines.append("")

    # ---- step statistics -------------------------------------------------
    lines.append("## Step-time statistics")
    lines.append("")
    if steps:
        dts = [s["dt_ms"] for s in steps]
        toks = [s["tok_per_s"] for s in steps]
        losses = [s["loss"] for s in steps]
        st = _summary_stats(dts)
        tt = _summary_stats(toks)
        lines.append(
            _table(
                ["metric", "min", "median", "mean", "max"],
                [
                    ["step time (ms)", f"{st['min']:.1f}", f"{st['median']:.1f}", f"{st['mean']:.1f}", f"{st['max']:.1f}"],
                    ["tokens/s", f"{tt['min']:,}", f"{tt['median']:,.0f}", f"{tt['mean']:,.0f}", f"{tt['max']:,}"],
                ],
            )
        )
        lines.append("")
        lines.append(f"- Steps observed: **{len(steps)}** (step {steps[0]['step']} → {steps[-1]['step']}).")
        lines.append(f"- Loss: **{losses[0]:.4f} → {losses[-1]:.4f}** (Δ {losses[0] - losses[-1]:+.4f}).")
        lines.append("")
    else:
        lines.append("_No per-step lines found in `train.log`._")
        lines.append("")

    # ---- top compute -----------------------------------------------------
    lines.append("## Top operators by self time")
    lines.append("")
    if has_dev_time:
        lines.append("Sorted by CUDA (device) self time.")
        lines.append("")
        top = by_dev[:15]
        rows_tbl = [
            [
                i + 1,
                f"`{r['key']}`",
                f"{r.get('count', 0):,}",
                _us(r.get("self_device_time_total_us", 0)),
                f"{100.0 * r.get('self_device_time_total_us', 0) / total_dev:.1f}%",
            ]
            for i, r in enumerate(top)
        ]
        lines.append(_table(["#", "operator", "calls", "self CUDA", "% CUDA"], rows_tbl))
    else:
        lines.append(
            "Sorted by CPU self time. **CUDA kernel time is unavailable on this host** "
            "(see Limitations), so this reflects host-side operator dispatch cost."
        )
        lines.append("")
        top = by_cpu[:15]
        rows_tbl = [
            [
                i + 1,
                f"`{r['key']}`",
                f"{r.get('count', 0):,}",
                _us(r.get("self_cpu_time_total_us", 0)),
                f"{100.0 * r.get('self_cpu_time_total_us', 0) / total_cpu:.1f}%",
            ]
            for i, r in enumerate(top)
        ]
        lines.append(_table(["#", "operator", "calls", "self CPU", "% CPU"], rows_tbl))
    lines.append("")
    lines.append(
        f"Total host-side operator self time: **{_us(total_cpu)}**"
        + (f"; total device self time: **{_us(total_dev)}**." if has_dev_time else ".")
    )
    lines.append("")

    # ---- category breakdown ---------------------------------------------
    lines.append("## Operator-category breakdown (host-side self time)")
    lines.append("")
    cat_rows = [
        [c, _us(v), f"{100.0 * v / total_cpu:.1f}%"]
        for c, v in sorted(cats.items(), key=lambda kv: kv[1], reverse=True)
    ]
    lines.append(_table(["category", "self CPU", "share"], cat_rows))
    lines.append("")

    # ---- memory ----------------------------------------------------------
    lines.append("## Device memory by operator")
    lines.append("")
    if total_dev_mem > 0:
        topm = [r for r in by_mem if r.get("self_device_memory_usage", 0) > 0][:12]
        rows_tbl = [
            [i + 1, f"`{r['key']}`", f"{r.get('count', 0):,}", _bytes(r.get("self_device_memory_usage", 0))]
            for i, r in enumerate(topm)
        ]
        lines.append(_table(["#", "operator", "calls", "device mem (cumulative)"], rows_tbl))
    else:
        lines.append("_No device-memory events captured._")
    lines.append("")

    # ---- observations ----------------------------------------------------
    lines.append("## Observations & recommendations")
    lines.append("")
    obs: list[str] = []

    if steps:
        step_wall_s = sum(s["dt_ms"] for s in steps) / 1000.0
        host_s = total_cpu / 1e6
        frac = host_s / step_wall_s if step_wall_s else 0.0
        obs.append(
            f"**Host vs device.** {host_s:.1f}s of host-side operator time was recorded over "
            f"~{step_wall_s:.1f}s of measured step wall time ({frac * 100:.0f}%). CUDA kernel time is not "
            "available on this host, so this cannot cleanly separate CPU work from GPU execution; a value "
            "near 100% means the host was continuously busy, which on this eager, non-compiled build points "
            "at launch/synchronization overhead in addition to genuine compute."
        )

    if has_dev_time:
        top1 = by_dev[0]
        obs.append(
            f"**Dominant kernel.** `{top1['key']}` accounts for "
            f"{100.0 * top1.get('self_device_time_total_us', 0) / total_dev:.1f}% of CUDA self time."
        )
    else:
        top1 = by_cpu[0] if by_cpu else None
        if top1 is not None:
            obs.append(
                f"**Largest host-side op.** `{top1['key']}` ({top1.get('count', 0):,} calls, "
                f"{_us(top1.get('self_cpu_time_total_us', 0))}). On this host this is a proxy for *what* "
                "runs, not for how long the GPU spends on it."
            )

    sync_rows = [r for r in rows if "local_scalar_dense" in r.get("key", "")]
    if sync_rows:
        sync_us = sum(r.get("self_cpu_time_total_us", 0.0) for r in sync_rows)
        sync_calls = sum(r.get("count", 0) for r in sync_rows)
        obs.append(
            f"**Device-to-host synchronizations.** `aten::_local_scalar_dense` ran {sync_calls:,}x "
            f"({_us(sync_us)}, {100.0 * sync_us / total_cpu:.1f}% of host operator time). It is emitted by "
            "`.item()` / Python-scalar conversions and forces the GPU pipeline to drain. Removing such "
            "conversions from the step loop (e.g. logging/scheduling) is usually a cheap win."
        )

    emb_rows = [r for r in rows if "embedding_dense_backward" in r.get("key", "")]
    if emb_rows:
        emb_us = sum(r.get("self_cpu_time_total_us", 0.0) for r in emb_rows)
        obs.append(
            f"**Embedding backward.** `aten::embedding_dense_backward` is {100.0 * emb_us / total_cpu:.1f}% of "
            "host operator time; the value-embedding backward path is worth inspecting if you want to cut "
            "host cost."
        )

    if "attention" in cats and total_cpu:
        obs.append(
            f"**Attention.** {100.0 * cats['attention'] / total_cpu:.1f}% of *host* operator time is "
            "attention-family ops (dispatch cost only). On Pascal the SDPA math backend materializes the "
            "full score matrix, so real GPU attention cost grows as O(seq^2) -- the interval is not visible "
            "in this trace, but it is why longer `--max-seq-len` gets expensive quickly."
        )
    if "matmul/linear" in cats and total_cpu:
        obs.append(
            f"**Matmul/linear.** {100.0 * cats['matmul/linear'] / total_cpu:.1f}% of host operator time; the "
            "1080 Ti has no tensor cores and runs fp32, so GEMMs are the throughput ceiling."
        )
    if wsl:
        obs.append(
            "**TorchInductor disabled.** `TORCH_COMPILE_DISABLE=1` is required on Pascal (Triton needs "
            "SM 70+); on newer hardware, enabling it would fuse elementwise ops and cut host dispatch."
        )

    for i, o in enumerate(obs, 1):
        lines.append(f"{i}. {o}")
    lines.append("")

    # ---- limitations -----------------------------------------------------
    lines.append("## Limitations")
    lines.append("")
    if not has_dev_time:
        lines.append(
            "- **No CUDA kernel timings.** CUPTI cannot initialize under this WSL2 setup "
            "(`CUPTI_ERROR_NOT_INITIALIZED`), so `torch.profiler`, Nsight Systems, and Nsight Compute cannot "
            "attribute time to individual GPU kernels. The report therefore uses host-side operator time and "
            "device-memory events; the per-operator numbers are *dispatch* costs, not GPU execution times."
        )
        lines.append(
            "- To obtain true kernel-level profiles, re-run the identical command on native Linux (non-WSL) "
            "with an NVIDIA driver that supports CUPTI. `scripts/profile_train.py` automatically records CUDA "
            "activities when they are available."
        )
    lines.append(
        "- Pure-Python work (e.g. the RustBPE tokenizer and dataloader bookkeeping) is not emitted as ATen "
        "operators and so does not appear in `key_averages`; use `--with-stack` or a flamegraph to see it."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Chrome/Perfetto trace: `{os.path.join(profile_dir, 'chrome_trace.json.gz')}`")
    lines.append(f"- TensorBoard trace: `{os.path.join(profile_dir, 'tensorboard')}/`")
    lines.append(f"- Raw operator table: `{ka_path}`")
    lines.append(f"- Trainer stdout: `{os.path.join(profile_dir, 'train.log')}`")
    lines.append("")

    report = "\n".join(lines)
    out_path = out_path or os.path.join(profile_dir, "PROFILE_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate a Markdown report from a nanochat profile")
    ap.add_argument("--dir", required=True, help="profile directory produced by scripts/profile_train.py")
    ap.add_argument("--out", default=None, help="output Markdown path (default: <dir>/PROFILE_REPORT.md)")
    args = ap.parse_args()
    path = generate_report(args.dir, args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

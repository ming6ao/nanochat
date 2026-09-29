# Running nanochat on the local GTX 1080 Ti (WSL2)

This document summarizes the **working** local setup, the commands to run a short
100-step training job, and how to capture a PyTorch profile and report.

The card is a Pascal **GTX 1080 Ti (SM 6.1, 11 GB)** under **WSL2** with a 536.99
driver (CUDA 12.2). It cannot use bf16, fp8, tensor cores, FlashAttention-3, or
`torch.compile`/Triton. The setup below works around all of that.

---

## 1. Environment (already provisioned)

```bash
cd ~/repos/nanochat
source .venv/bin/activate

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export TORCH_COMPILE_DISABLE=1     # Triton does not support Pascal
export NANOCHAT_DTYPE=float32      # bf16 needs SM 80+
export WANDB_RUN=dummy             # no wandb login
```

Why these choices:

| Piece | Value | Reason |
|---|---|---|
| PyTorch | **2.7.1+cu118** | CUDA 11.8 runtime is ≤ the WSL cap (12.2) and ships `sm_60` kernels (run on `sm_61`) |
| `TORCH_COMPILE_DISABLE` | `1` | Inductor/Triton requires Volta+ (`sm_70`) |
| `NANOCHAT_DTYPE` | `float32` | auto-selected on SM < 80 anyway |
| attention | SDPA math fallback | FA3 is Hopper/Ada/Ampere only |
| `--window-pattern` | `L` | SDPA has no efficient sliding-window path; the code warns otherwise |

> The one required code fix lives in `nanochat/common.py::compute_init`
> (a WSL/pre-Ampere workaround that flushes a stale `cudaErrorNotSupported` left
> by the dtype autodetect). Without it the first CUDA allocation aborts.

Data and tokenizer are already in `$NANOCHAT_BASE_DIR` (`base_data_climbmix/`,
`tokenizer/`). Re-download more shards with `python -m nanochat.dataset -n <N>`;
re-train the tokenizer only if the data changes.

---

## 2. Run 100 training steps (no eval)

```bash
cd ~/repos/nanochat && source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=float32 WANDB_RUN=dummy

python -m scripts.base_train \
    --depth=4 --window-pattern=L \
    --max-seq-len=512 --device-batch-size=8 --total-batch-size=4096 \
    --num-iterations=100 --warmup-steps=3 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --save-every=-1 \
    --run=dummy --model-tag=demo100
```

* `--device-batch-size` sets peak VRAM; `--total-batch-size` controls gradient
  accumulation. `total_batch_size` must be a multiple of
  `device_batch_size × max_seq_len`.
* Eval is disabled with `-1` on `--eval-every`, `--core-metric-every`, and
  `--sample-every`.
* The trainer always writes one checkpoint at the final step, to
  `$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>/`.

---

## 3. Profile the training

`scripts/profile_train.py` wraps the real trainer in `torch.profiler`, forwards
everything after `--` to `scripts.base_train`, and saves the trace plus a
Markdown report.

```bash
cd ~/repos/nanochat && source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=float32 WANDB_RUN=dummy

python -m scripts.profile_train --steps 100 --out profiles/d8_s512 -- \
    --depth=8 --window-pattern=L \
    --max-seq-len=512 --device-batch-size=8 --total-batch-size=4096 \
    --warmup-steps=3 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --save-every=-1 --run=dummy --model-tag=prof_d8
```

Artifacts written to `profiles/d8_s512/` (git-ignored):

| File | What it is |
|---|---|
| `chrome_trace.json.gz` | gzipped Chrome/Perfetto trace — open at <chrome://tracing> or <https://ui.perfetto.dev> |
| `tensorboard/` | `*.pt.trace.json.gz` for `tensorboard --logdir` |
| `key_averages.json` | per-operator timings, call counts, memory |
| `train.log` | captured trainer stdout (loss, dt, tok/s, peak memory) |
| `summary.json` | run metadata (config, wall time, whether CUDA was captured) |
| `PROFILE_REPORT.md` | generated insights report |

Regenerate the report at any time:

```bash
python -m scripts.profile_report --dir profiles/d8_s512
```

### Important: CUPTI is unavailable under WSL2

`torch.profiler` cannot initialize CUPTI on this host
(`CUPTI_ERROR_NOT_INITIALIZED`), so **GPU kernel timings are not captured** by
`torch.profiler`, Nsight Systems, or Nsight Compute. The profile still records:

* host-side operator dispatch times and call counts,
* device-memory allocation events per operator,
* per-step wall-clock time / throughput (from `train.log`).

The same `scripts/profile_train.py` command captures CUDA kernel activities
automatically when run on **native Linux** (not WSL). For true kernel-level
profiling of this model, re-run there.

---

## 4. Scaling tiers (measured on this machine)

Measured frontier (5-step probes, `--window-pattern L`):

| depth | seq | device batch | peak VRAM | tokens/s | note |
|---|---|---|---|---|---|
| 6 | 512 | 8 | 3.6 GB | ~18,300 | fast iteration |
| 8 | 1024 | 8 | 8.5 GB | ~14,000 | best value |
| 10 | 1024 | 4 | 6.1 GB | ~6,000 | comfortable |
| 12 | 512 | 8 | 7.9 GB | ~4,600 | largest reliable |
| 12 | 1024 | 4 | 7.9 GB | ~3,450 | largest reliable |
| 14+ | 512 | 8 | ≥10 GB | ≤730 | thrashes / over VRAM |

Compute-optimal horizons use `--target-param-data-ratio=12` (the nanochat
default). Let the trainer derive the horizon and batch:

```bash
# Example: depth 10, 1024 ctx, auto batch + auto iteration count
python -m scripts.base_train \
    --depth=10 --window-pattern=L \
    --max-seq-len=1024 --device-batch-size=4 \
    --total-batch-size=-1 --target-param-data-ratio=12 --num-iterations=-1 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --save-every=500 \
    --run=dummy --model-tag=d10_s1024
```

| tier | depth | seq | dev batch | optimal tokens | shards (`-n`) | rough time |
|---|---|---|---|---|---|---|
| quick | 6 | 512 | 8 | 278M | 6 | ~4 h |
| overnight | 8 | 1024 | 8 | 503M | 10 | ~10 h |
| weekend | 10 | 1024 | 4 | 841M | 17 | ~40 h |
| max fit | 12 | 512 | 8 | 1.32B | 26 | ~79 h |

Resume an interrupted run with `--resume-from-step N --model-tag <tag>`.

---

## 5. Quick reference

```bash
# env
cd ~/repos/nanochat && source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=float32 WANDB_RUN=dummy

# 100 steps
python -m scripts.base_train --depth=4 --window-pattern=L --max-seq-len=512 \
    --device-batch-size=8 --total-batch-size=4096 --num-iterations=100 --warmup-steps=3 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
    --run=dummy --model-tag=demo100

# profile 100 steps + report
python -m scripts.profile_train --steps 100 --out profiles/d8_s512 -- \
    --depth=8 --window-pattern=L --max-seq-len=512 --device-batch-size=8 \
    --total-batch-size=4096 --warmup-steps=3 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --save-every=-1 --run=dummy --model-tag=prof_d8

# regenerate a report
python -m scripts.profile_report --dir profiles/d8_s512
```

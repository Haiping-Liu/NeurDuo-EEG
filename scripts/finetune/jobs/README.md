# Fine-tuning jobs

`finetune.sh` is the one Slurm script for every NeurDuo fine-tuning run. A run is five folds, submitted
as a job array with one GPU per fold. The settings are `KEY=VALUE` arguments; the header of the script
lists them all.

```bash
# print the sbatch command, the run directory and the job name; nothing is submitted
scripts/finetune/jobs/finetune.sh SIZE=small VARIANT=no_memory SCHED=sched4 SEED=0
# submit
scripts/finetune/jobs/finetune.sh SIZE=small VARIANT=no_memory SCHED=sched4 SEED=0 --submit
# smoke test: one fold, two epochs, its own run directory
scripts/finetune/jobs/finetune.sh FOLDS=0 TIME=01:00:00 TAG=_smoke EXTRA="--epochs 2 --warmup_epochs 1" --submit
```

| Setting | Values (default first) | Meaning |
|---|---|---|
| `DATASET` | `chbmit`, `chbmit_2ch`, `sleepedf` | `chbmit_2ch` is T7 and P8 of the 15-channel data |
| `SIZE` | `small`, `base`, `large` | `ckpt/release/neurduo_eeg_<SIZE>.pt` |
| `VARIANT` | `full`, `no_memory`, `no_slow_memory`, `no_slow_stream`, `no_channel_attention`, `no_channel_tokens`, `pure_mamba`, `pure_mamba_scratch` | Small only. `ckpt/ablation/small_<VARIANT>.pt`; `pure_mamba_scratch` starts from random weights |
| `SCHED` | `normal`, `sched4`, `warm12` | learning-rate curve, see below |
| `READOUT` | `flatten`, `pool` | head: all tokens flattened (24.8M parameters) or mean over tokens (1.9M) |
| `HEAD_LR` | empty, or a value such as `1e-4` | empty is 7.9e-4, eight times the backbone's 1e-4 |
| `SEED` | `42` | fine-tuning seed; the folds do not depend on it |
| `FOLDS` | `0-4` | Slurm array spec, e.g. `FOLDS=0` reruns one fold |
| `TAG`, `EXTRA`, `TIME` | | run-directory suffix, extra `run_finetune.py` arguments, wall-time limit |

Results go to `/projects/u6xc/haiping/neurduo/runs/finetune/<NAME>/<DATASET>/`, where
`NAME = <SIZE>[_<VARIANT>][_<SCHED>][_poolhead][_hlr<HEAD_LR>][TAG]`: one `fold{k}_seed<SEED>.json` per
fold, `predictions/` (test and validation of the selected epoch), `models/` (the selected epoch),
`models/epochs/` (the model after every epoch, `fold{k}_seed<SEED>_ep<EE>.pt`, same format) and
`checkpoints/` (the last epoch with the optimiser state, for resuming). The per-epoch models take
about 0.5 GB per fold with the small head and 2.4 GB with the paper's head for Small, 4 GB for Base
and 7.4 GB for Large. Logs go to `/projects/u6xc/haiping/neurduo/logs/finetune/<DATASET>/`. A fold whose result file
exists is skipped when the job is submitted again (`export FORCE=1` reruns it); an unfinished fold
continues from its last-epoch checkpoint.

Aggregate a run with
`python scripts/finetune/aggregate.py --dataset <DATASET> --output_dir <run directory without the dataset part>`.

## What every run shares

- CHB-MIT: 20 x 30 s units (10 min of context), state carried across units, whole-sequence
  normalisation, 5 case-level folds, model chosen by validation AUC-PR, 20 epochs.
- Sleep-EDF: Fpz and Pz, 20 x 30 s units, state carried across units, per-unit normalisation, 5 subject
  folds, model chosen by validation kappa, 20 epochs. The paper's Sleep-EDF rows use per-unit
  normalisation; only CHB-MIT uses whole-sequence normalisation.
- One step takes 8 sequences (160 units) at once on one GPU, with no gradient accumulation. The paper
  runs split each step into 4 accumulated micro-batches for memory; the effective batch is the same.
- The job sets its own `TMPDIR` under `/tmp`. The default one is removed whenever another job of the
  same user on the node ends, which kills the data-loader workers.

## Learning-rate curves

- `normal`: 3 warm-up epochs, then cosine decay to zero. The current code's default.
- `sched4` (`--sched_stretch 4`): reproduces the curve of the paper's CHB-MIT runs (harness_v20, August
  2026). Those used `--grad_accum 4` with a scheduler that counted micro-batches, so warm-up lasted
  until epoch 12 of 20 and the learning rate was still 9.7e-5 at the end.
- `warm12` (`--warmup_epochs 12`): the same first 12 epochs as `sched4`, then cosine decay to zero over
  the last 8. Exploratory.

## Earlier scripts

`finetune.sh` replaces the per-run scripts that used to sit in `scripts/finetune/`. The run directories
are unchanged, so earlier results line up with new ones.

| Earlier script | Settings now |
|---|---|
| `chbmit_small.sh`, `chbmit_base.sh`, `chbmit_large.sh` | `SIZE=small` / `base` / `large` |
| `chbmit_small_sched4.sh`, `chbmit_base_sched4.sh`, `chbmit_large_sched4.sh` | `SIZE=... SCHED=sched4` |
| `chbmit_warm12.sh` | `SIZE=... SCHED=warm12` |
| `chbmit_small_no_memory.sh`, `chbmit_small_no_slow_memory.sh`, `chbmit_small_no_slow_stream.sh` | `VARIANT=no_memory` / `no_slow_memory` / `no_slow_stream` |
| `chbmit_small_no_slow_stream_sched4.sh`, `chbmit_small_ablation_sched4.sh` | `VARIANT=... SCHED=sched4` |
| `chbmit_small_pure_mamba_scratch.sh` | `VARIANT=pure_mamba_scratch` |
| `chbmit_2ch_small.sh`, `chbmit_2ch_base.sh`, `chbmit_2ch_large.sh` | `DATASET=chbmit_2ch SIZE=...` |
| `chbmit_2ch_small_pure_mamba_scratch.sh` | `DATASET=chbmit_2ch VARIANT=pure_mamba_scratch` |
| `sleepedf_small_pure_mamba_scratch.sh` | `DATASET=sleepedf VARIANT=pure_mamba_scratch` |
| `chbmit_headlr.sh` (never submitted) | `HEAD_LR=1e-4`, with `READOUT=pool` for the small head |

Two differences from the earlier scripts: every run now keeps the model of every epoch, and the job
name always ends in the seed (`ft-chbmit-small-sched4-s42`), so logs of new runs carry `-s<SEED>`.

`VARIANT=no_channel_tokens` does not load at present: the checkpoint has no `channel_attn` weights and
the current model code expects them.

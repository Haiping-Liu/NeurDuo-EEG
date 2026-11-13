#!/bin/bash -l
# One Slurm job script for every NeurDuo fine-tuning run (5 folds as a job array, one GPU per fold).
# It replaces the per-run scripts chbmit_*.sh and sleepedf_*.sh; README.md in this folder maps the old
# script names to the new settings and keeps the notes on each protocol.
#
# Print the sbatch command (nothing is submitted):
#     scripts/finetune/jobs/finetune.sh SIZE=small VARIANT=no_memory SCHED=sched4 SEED=0
# Submit it:
#     scripts/finetune/jobs/finetune.sh SIZE=small VARIANT=no_memory SCHED=sched4 SEED=0 --submit
#
# Settings (KEY=VALUE, any order; defaults first):
#   DATASET  chbmit | chbmit_2ch | sleepedf
#   SIZE     small | base | large                    backbone ckpt/release/neurduo_eeg_<SIZE>.pt
#   VARIANT  full | no_memory | no_slow_memory | no_slow_stream | no_channel_attention |
#            no_channel_tokens | pure_mamba          ckpt/ablation/small_<VARIANT>.pt (Small only)
#            pure_mamba_scratch                      configs/pretrain/small_pure_mamba.json, random init
#   SCHED    normal   3 warm-up epochs, cosine decay to zero over 20 epochs
#            sched4   --sched_stretch 4: warm-up until epoch 12, almost no decay (the paper-era curve)
#            warm12   --warmup_epochs 12: warm-up until epoch 12, cosine decay to zero over the last 8
#   READOUT  flatten  the paper's head (all tokens flattened, about 24.8M parameters)
#            pool     mean over tokens (about 1.9M parameters)
#   HEAD_LR  empty = 1e-3 * sqrt(160/256) = 7.9e-4 (8x the backbone's 1e-4) | a value, e.g. 1e-4
#   SEED     42       fine-tuning seed; the folds are the same for every seed
#   FOLDS    0-4      Slurm array spec, e.g. FOLDS=0 to rerun one fold
#   TAG      suffix for the run directory, e.g. TAG=_smoke
#   EXTRA    further run_finetune.py arguments, e.g. EXTRA="--epochs 2 --warmup_epochs 1"
#   TIME     24:00:00 wall-time limit
# A fold whose result file exists is skipped (export FORCE=1 to rerun it); an unfinished fold continues
# from its last-epoch checkpoint.
#
# Run directory: /projects/u6xc/haiping/neurduo/runs/finetune/<NAME>/<DATASET>/ with
#     NAME = <SIZE>[_<VARIANT>][_<SCHED>][_poolhead][_hlr<HEAD_LR>][TAG]
#     fold{k}_seed<SEED>.json, predictions/, models/fold{k}_seed<SEED>.pt (the selected epoch),
#     models/epochs/fold{k}_seed<SEED>_ep<EE>.pt (every epoch), checkpoints/ (last epoch with optimiser)
# Log: /projects/u6xc/haiping/neurduo/logs/finetune/<DATASET>/ft-<DATASET>-<NAME>-s<SEED>-<job>_<fold>.log
#SBATCH --job-name=ft
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
set -eo pipefail

DATASET=${DATASET:-chbmit}; SIZE=${SIZE:-small}; VARIANT=${VARIANT:-full}; SCHED=${SCHED:-normal}
READOUT=${READOUT:-flatten}; HEAD_LR=${HEAD_LR:-}; SEED=${SEED:-42}; FOLDS=${FOLDS:-0-4}
TAG=${TAG:-}; EXTRA=${EXTRA:-}; TIME=${TIME:-24:00:00}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
ROOT=${ROOT:-/projects/u6xc/haiping/neurduo}
SUBMIT=0

die() { echo "finetune.sh: $*" >&2; exit 1; }

# Command-line settings (only when started by hand; inside a job the settings come from the environment).
for a in "$@"; do
    case "$a" in
        --submit) SUBMIT=1 ;;
        DATASET=*|SIZE=*|VARIANT=*|SCHED=*|READOUT=*|HEAD_LR=*|SEED=*|FOLDS=*|TAG=*|EXTRA=*|TIME=*)
            printf -v "${a%%=*}" '%s' "${a#*=}" ;;
        *) die "unknown argument '$a' (see the header of this file)" ;;
    esac
done

case "$DATASET" in chbmit|chbmit_2ch|sleepedf) ;; *) die "DATASET=$DATASET" ;; esac
case "$SIZE" in small|base|large) ;; *) die "SIZE=$SIZE" ;; esac
case "$VARIANT" in
    full) ;;
    no_memory|no_slow_memory|no_slow_stream|no_channel_attention|no_channel_tokens|pure_mamba|pure_mamba_scratch)
        [ "$SIZE" = small ] || die "VARIANT=$VARIANT exists for SIZE=small only" ;;
    *) die "VARIANT=$VARIANT" ;;
esac
case "$SCHED" in
    normal) SCHED_ARGS="" ;;
    sched4) SCHED_ARGS="--sched_stretch 4" ;;
    warm12) SCHED_ARGS="--warmup_epochs 12" ;;
    *) die "SCHED=$SCHED" ;;
esac
case "$READOUT" in flatten|pool) ;; *) die "READOUT=$READOUT" ;; esac

NAME=$SIZE
if [ "$VARIANT" != full ]; then NAME=${NAME}_$VARIANT; fi
if [ "$SCHED" != normal ]; then NAME=${NAME}_$SCHED; fi
if [ "$READOUT" = pool ]; then NAME=${NAME}_poolhead; fi
if [ -n "$HEAD_LR" ]; then NAME=${NAME}_hlr$HEAD_LR; fi
NAME=$NAME$TAG
OUT=$ROOT/runs/finetune/$NAME

if [ "$VARIANT" = pure_mamba_scratch ]; then
    INIT="--no_pretrained --model_config configs/pretrain/small_pure_mamba.json"
else
    if [ "$VARIANT" = full ]; then CKPT=$REPO/ckpt/release/neurduo_eeg_$SIZE.pt
    else CKPT=$REPO/ckpt/ablation/small_$VARIANT.pt; fi
    [ -f "$CKPT" ] || die "no checkpoint $CKPT"
    INIT="--checkpoint $CKPT"
fi
case "$DATASET" in chbmit_2ch) DATA=$REPO/finetune_data/chbmit ;; *) DATA=$REPO/finetune_data/$DATASET ;; esac

# ---- started by hand: print or submit the sbatch command -------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
    JOB=ft-$DATASET-${NAME//_/-}-s$SEED
    LOG=$ROOT/logs/finetune/$DATASET/%x-%A_%a.log
    echo "run dir: $OUT/$DATASET/"
    echo "sbatch -J $JOB --array=$FOLDS --time=$TIME --output=$LOG $SELF"
    if [ "$SUBMIT" = 1 ]; then
        export DATASET SIZE VARIANT SCHED READOUT HEAD_LR SEED TAG EXTRA REPO ROOT FORCE
        sbatch -J "$JOB" --array="$FOLDS" --time="$TIME" --output="$LOG" "$SELF"
    else
        echo "(not submitted; add --submit)"
    fi
    exit 0
fi

# ---- inside the job --------------------------------------------------------------------------------
FOLD=$SLURM_ARRAY_TASK_ID
RESULT=$OUT/$DATASET/fold${FOLD}_seed$SEED.json
if [ -f "$RESULT" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "$RESULT exists, nothing to do (FORCE=1 reruns the fold)"; exit 0
fi
ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"
# The default TMPDIR (/local/user/<uid> -> /run/user/<uid>) is removed whenever another of the user's jobs
# on the same node ends, which kills the data-loader workers; use a directory of this job's own.
export TMPDIR=$(mktemp -d /tmp/neurduo-${SLURM_JOB_ID:-$$}-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${FOLD} name=$NAME dataset=$DATASET size=$SIZE variant=$VARIANT sched=$SCHED readout=$READOUT head_lr=${HEAD_LR:-default} seed=$SEED commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
git -C "$REPO" status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

HEAD_ARGS="--readout $READOUT"
if [ -n "$HEAD_LR" ]; then HEAD_ARGS="$HEAD_ARGS --head_lr $HEAD_LR"; fi

cd "$REPO"
python scripts/finetune/run_finetune.py --dataset "$DATASET" --data_dir "$DATA" $INIT \
    --output_dir "$OUT" --fold "$FOLD" --seed "$SEED" \
    --resume --keep_ckpt --save_all_epochs --model_dir "$OUT/$DATASET/models" $SCHED_ARGS $HEAD_ARGS $EXTRA
echo "done $(date '+%F %T')"

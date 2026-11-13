#!/bin/bash -l
# Score other epochs of a finished fine-tuning run on its validation and test folds, without retraining
# (scripts/finetune/eval_epochs.py). One array task per fold, a few minutes each.
#
# Usage: RUN=small_no_memory_sched4 [SEED=42] [SOURCE=checkpoint|epochs] [DATASET=chbmit] \
#            sbatch -J ev-small-no-memory-sched4-s42 scripts/finetune/jobs/eval_epochs.sh
#   SOURCE=checkpoint  the last epoch kept in checkpoints/, and the selected epoch again as a check
#   SOURCE=epochs      every epoch saved in models/epochs/ by --save_all_epochs
#   output: /projects/u6xc/haiping/neurduo/runs/finetune/<RUN>/<DATASET>/epoch_eval/
#   log:    /projects/u6xc/haiping/neurduo/logs/finetune/epoch_eval/
#SBATCH --job-name=ev
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/finetune/epoch_eval/%x-%A_%a.log
set -eo pipefail

: "${RUN:?set RUN, the run directory name, e.g. RUN=small_sched4}"
SEED=${SEED:-42}; SOURCE=${SOURCE:-checkpoint}; DATASET=${DATASET:-chbmit}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
ROOT=${ROOT:-/projects/u6xc/haiping/neurduo}
ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
case "$DATASET" in chbmit_2ch) DATA=$REPO/finetune_data/chbmit ;; *) DATA=$REPO/finetune_data/$DATASET ;; esac
FOLD=$SLURM_ARRAY_TASK_ID
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"
# A directory of this job's own: the default TMPDIR is removed when another job of the user ends.
export TMPDIR=$(mktemp -d /tmp/neurduo-${SLURM_JOB_ID:-$$}-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${FOLD} run=$RUN seed=$SEED source=$SOURCE commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
cd "$REPO"
python scripts/finetune/eval_epochs.py --run_dir "$ROOT/runs/finetune/$RUN" --dataset "$DATASET" \
    --data_dir "$DATA" --fold "$FOLD" --seed "$SEED" --source "$SOURCE"
echo "done $(date '+%F %T')"

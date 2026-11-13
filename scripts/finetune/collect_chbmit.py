"""Collect the CHB-MIT fine-tuning runs into CSV tables.

  folds.csv   one row per run, seed and fold: the selected epoch with its validation and test metrics,
              and, where epoch_eval/ exists (scripts/finetune/eval_epochs.py), the last epoch and the
              mean of the two epochs' scores
  runs.csv    one row per run and seed: mean and standard deviation over the folds, pooled FP/h
  curves.csv  one row per run, seed, fold and epoch, read from the training logs: training loss,
              validation metrics and backbone learning rate

Usage: python scripts/finetune/collect_chbmit.py [--runs DIR] [--logs DIR] [--out DIR]
"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

ROOT = '/projects/u6xc/haiping/neurduo'
VARIANTS = ['pure_mamba_scratch', 'no_slow_memory', 'no_slow_stream', 'no_channel_attention',
            'no_channel_tokens', 'no_memory', 'pure_mamba']
EPOCH = re.compile(r"Epoch (\d+): loss ([\d.]+) \| val auroc ([\d.]+) auc_pr ([\d.]+) "
                   r"fp_per_hour ([\d.]+) \| lr ([\d.]+)")
SAVED = re.compile(r"saved test predictions: \S*/runs/finetune/([^/]+)/([^/]+)/predictions/"
                   r"fold(\d+)_seed(\d+)\.npz")


def describe(run):
    """Split a run directory name into size, variant, curve, head and head learning rate."""
    size, rest = run.split('_', 1) if '_' in run else (run, '')
    variant = next((v for v in VARIANTS if rest.startswith(v)), 'full')
    sched = 'sched4' if 'sched4' in rest else 'warm12' if 'warm12' in rest else 'normal'
    readout = 'pool' if 'poolhead' in rest else 'flatten'
    m = re.search(r'hlr([^_]+)', rest)
    return dict(size=size, variant=variant, sched=sched, readout=readout,
                head_lr=m.group(1) if m else '')


def auc_pr(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(r, p))


def write(path, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'{path}: {len(rows)} rows')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default=f'{ROOT}/runs/finetune')
    ap.add_argument('--logs', default=f'{ROOT}/logs/finetune')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'results' / 'chbmit'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    folds = []
    for p in sorted(glob.glob(f'{args.runs}/*/chbmit*/fold*_seed*.json')):
        run, dataset = p.split('/')[-3:-1]
        j = json.load(open(p))
        row = dict(run=run, dataset=dataset, **describe(run), seed=j['seed'], fold=j['fold'],
                   selected_epoch=j['best_epoch'])
        for k in ('val_auc_pr', 'val_auroc', 'val_fp_per_hour', 'val_n_positive', 'val_hours',
                  'test_auc_pr', 'test_auroc', 'test_fp_per_hour', 'test_n_positive', 'test_hours',
                  'test_fp_per_hour_valthr_sens50', 'test_sensitivity_valthr_sens50',
                  'training_time_min', 'n_params_backbone', 'n_params_head'):
            row[k] = j.get(k, '')
        stem = f"fold{j['fold']}_seed{j['seed']}"
        e = f'{os.path.dirname(p)}/epoch_eval/{stem}'
        if os.path.exists(e + '.json'):
            last = json.load(open(e + '.json'))['epochs']['last']
            a, b = np.load(e + '_best.npz'), np.load(e + '_last.npz')
            row.update(last_epoch=last['epoch'], last_val_auc_pr=last['val_auc_pr'],
                       last_test_auc_pr=last['test_auc_pr'],
                       last_test_fp_per_hour=last['test_fp_per_hour'],
                       mean2_test_auc_pr=auc_pr(a['test_y_true'],
                                                (a['test_y_score'] + b['test_y_score']) / 2))
        folds.append(row)
    write(f'{args.out}/folds.csv', folds)

    runs = []
    keys = sorted({(r['run'], r['dataset'], r['seed']) for r in folds})
    for run, dataset, seed in keys:
        rs = [r for r in folds if (r['run'], r['dataset'], r['seed']) == (run, dataset, seed)]
        t = np.array([r['test_auc_pr'] for r in rs])
        v = np.array([r['val_auc_pr'] for r in rs])
        hours = np.array([r['test_hours'] for r in rs])
        fa = np.array([r['test_fp_per_hour'] for r in rs]) * hours
        row = dict(run=run, dataset=dataset, **describe(run), seed=seed, n_folds=len(rs),
                   test_auc_pr_mean=t.mean(), test_auc_pr_sd=t.std(),
                   test_fp_per_hour_pooled=fa.sum() / hours.sum(), val_auc_pr_mean=v.mean())
        if all('last_test_auc_pr' in r for r in rs):
            row.update(last_test_auc_pr_mean=np.mean([r['last_test_auc_pr'] for r in rs]),
                       mean2_test_auc_pr_mean=np.mean([r['mean2_test_auc_pr'] for r in rs]))
        runs.append(row)
    write(f'{args.out}/runs.csv', runs)

    best = {}
    for p in glob.glob(f'{args.logs}/chbmit*/*.log'):
        text = open(p, errors='ignore').read().replace('\r', '\n')
        m = SAVED.search(text)
        if not m:
            continue
        key = (m.group(1), m.group(2), int(m.group(4)), int(m.group(3)))
        ep = {int(a[0]): a for a in EPOCH.findall(text)}
        if key not in best or len(ep) > len(best[key][0]):
            best[key] = (ep, os.path.basename(p))
    curves = []
    for (run, dataset, seed, fold), (ep, log) in sorted(best.items()):
        for e in sorted(ep):
            _, loss, auroc, pr, fp, lr = ep[e]
            curves.append(dict(run=run, dataset=dataset, **describe(run), seed=seed, fold=fold,
                               epoch=e, train_loss=float(loss), val_auroc=float(auroc),
                               val_auc_pr=float(pr), val_fp_per_hour=float(fp),
                               backbone_lr=float(lr), log=log))
    write(f'{args.out}/curves.csv', curves)


if __name__ == '__main__':
    main()

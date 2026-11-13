"""Evaluate saved models of a finished sequence-task fine-tuning run on its validation and test folds.

A run reports the test score of one epoch only, the one with the best validation score. This script
scores other epochs of the same training run without retraining, to see how much the reported number
depends on which epoch was taken:

  --source checkpoint   checkpoints/<stem>.pt (kept by --keep_ckpt): the model after the last epoch
                        ('last') and the selected one again ('best', which must reproduce the test
                        score in the run's result file)
  --source epochs       models/epochs/<stem>_ep<EE>.pt (written by --save_all_epochs): every epoch

Writes <run_dir>/<dataset>/epoch_eval/<stem>.json (metrics per label) and <stem>_<label>.npz (the
validation and test scores of each unit).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_fm.config import EEGFMConfig
from eeg_fm.finetune.dataset import LoadDataset
from eeg_fm.finetune.dataset_specs import get_spec
from eeg_fm.finetune.net_eegfm import Model
from eeg_fm.finetune.net_seq import SeqModel, SingleEpochModel
from eeg_fm.finetune.trainer import _evaluator_cls


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_dir', required=True, help='e.g. .../runs/finetune/small_sched4')
    ap.add_argument('--dataset', default='chbmit')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--source', choices=['checkpoint', 'epochs'], default='checkpoint')
    ap.add_argument('--num_workers', type=int, default=8)
    args = ap.parse_args()

    res_dir = os.path.join(args.run_dir, args.dataset)
    stem = f'fold{args.fold}_seed{args.seed}'
    final = torch.load(f'{res_dir}/models/{stem}.pt', map_location='cpu', weights_only=False)
    a = final['args']
    ds = get_spec(args.dataset, args.data_dir)
    if a.get('context_len') is not None:
        ds['context_len'] = a['context_len']
    params = SimpleNamespace(
        dataset=args.dataset, task=ds['task'], num_of_classes=ds['n_outputs'],
        channels=ds['channels'], seg_sec=ds['seg_sec'], readout_budget=ds['readout_budget'],
        model_config=EEGFMConfig.from_dict(final['model_config']),
        use_pretrained_weights=False, foundation_dir=None, dropout=a['dropout'],
        context_len=ds.get('context_len', 1),
        cont_normalize=a.get('cont_normalize') or ds.get('cont_normalize', 'window'),
        readout=a.get('readout', 'flatten'), use_context_head=not a.get('no_context_head', False))

    states = {}
    if args.source == 'checkpoint':
        ck = torch.load(f'{res_dir}/checkpoints/{stem}.pt', map_location='cpu', weights_only=False)
        states['last'] = (ck['epoch'], ck['model'])
        states['best'] = (ck['best_epoch'], ck['best_model_states'])
    else:
        for p in sorted(glob.glob(f'{res_dir}/models/epochs/{stem}_ep*.pt')):
            e = torch.load(p, map_location='cpu', weights_only=False)
            states[f"ep{e['epoch']:02d}"] = (e['epoch'], e['model'])
    assert states, f'nothing to evaluate for {stem} in {res_dir}'

    sets = LoadDataset(ds, batch_size=8, fold=args.fold,
                       num_workers=args.num_workers)._load_sequences()
    bs = max(1, int(round(160 / params.context_len)))
    loaders = {s: DataLoader(sets[s], batch_size=bs, shuffle=False, num_workers=args.num_workers,
                             collate_fn=sets[s].collate) for s in ('val', 'test')}
    E = _evaluator_cls(params)
    evals = {s: E(params, loaders[s]) for s in loaders}

    model = Model(params)
    model = (SeqModel if params.context_len > 1 else SingleEpochModel)(model, params).cuda()
    out_dir = f'{res_dir}/epoch_eval'
    os.makedirs(out_dir, exist_ok=True)
    out = dict(run_dir=args.run_dir, dataset=args.dataset, fold=args.fold, seed=args.seed,
               source=args.source, epochs={})
    for label, (epoch, state) in states.items():
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            vm, _, vp = evals['val'].get_metrics_for_seq2seq(model, return_predictions=True)
            tm, _, tp = evals['test'].get_metrics_for_seq2seq(model, return_predictions=True)
        if hasattr(evals['test'], 'at_val_thresholds'):
            tm.update(evals['test'].at_val_thresholds(vp, tp))
        r = dict(epoch=int(epoch))
        r.update({f'val_{k}': v for k, v in vm.items()})
        r.update({f'test_{k}': v for k, v in tm.items()})
        out['epochs'][label] = r
        np.savez_compressed(f'{out_dir}/{stem}_{label}.npz', **{
            f'{split}_{k}': v for split, pr in (('val', vp), ('test', tp))
            for k, v in pr.items() if v is not None})
        key = 'auc_pr' if 'auc_pr' in tm else 'kappa'
        print(f"[{stem}] {label} (epoch {epoch}): val {key} {vm[key]:.5f} | test {key} {tm[key]:.5f}",
              flush=True)

    if args.source == 'checkpoint' and os.path.exists(f'{res_dir}/{stem}.json'):
        ref = json.load(open(f'{res_dir}/{stem}.json'))
        key = 'test_auc_pr' if 'test_auc_pr' in ref else 'test_kappa'
        diff = abs(out['epochs']['best'][key] - ref[key])
        out['best_reproduces_result'] = bool(diff < 1e-3)
        print(f"[{stem}] check: 'best' {key} {out['epochs']['best'][key]:.5f} vs result file "
              f"{ref[key]:.5f} (difference {diff:.2e})", flush=True)
    with open(f'{out_dir}/{stem}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', f'{out_dir}/{stem}.json', flush=True)


if __name__ == '__main__':
    main()

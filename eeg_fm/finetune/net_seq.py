"""Models for sequence tasks, which label every unit of a sequence of consecutive 30 s units."""

import torch.nn as nn

from eeg_fm.finetune.heads import StreamContextHead


class SeqModel(nn.Module):
    def __init__(self, base, params):
        super().__init__()
        self.base = base
        self.st_head = base.classifier
        self.context_len = params.context_len
        # Without the context head each unit is classified from its own readout, so anything from
        # earlier units has to come through the state the backbone carries.
        self.context = (StreamContextHead(params.num_of_classes, params.context_len,
                                          dropout=params.dropout)
                        if getattr(params, 'use_context_head', True) else None)

    def encode(self, x):
        b, l = x.shape[:2]
        return self.st_head.readout(self.base.tokens_continuous(x)).view(b, l, -1)

    def forward(self, x, valid_mask=None):
        h = self.encode(x)
        if self.context is None:
            return self.st_head.classify(h)
        return self.context(h, valid_mask)


class SingleEpochModel(nn.Module):
    def __init__(self, base, params):
        super().__init__()
        self.base = base

    def forward(self, x, valid_mask=None):
        b, l = x.shape[:2]
        assert l == 1, f'SingleEpochModel expects L=1, got {l}'
        return self.base(x.reshape(b, *x.shape[2:])).unsqueeze(1)

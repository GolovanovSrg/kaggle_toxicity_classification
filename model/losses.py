import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_labels, smoothing=0.0, ignore_index=-100, reduction="mean"):
        super().__init__()
        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing

        if smoothing > 0:
            if reduction == "mean":
                reduction = "batchmean"
            self.criterion = nn.KLDivLoss(reduction=reduction)
            n_ignore_idxs = 1 + (ignore_index >= 0)
            one_hot = torch.full(
                (1, n_labels), fill_value=(smoothing / (n_labels - n_ignore_idxs))
            )
            if ignore_index >= 0:
                one_hot[0, ignore_index] = 0
            self.register_buffer("one_hot", one_hot)
        else:
            self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        log_inputs = F.log_softmax(inputs, dim=-1)
        if self.confidence < 1:
            tdata = targets.data

            tmp = self.one_hot.repeat(targets.shape[0], 1)
            tmp.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if self.ignore_index >= 0:
                mask = torch.nonzero(tdata.eq(self.ignore_index)).squeeze(-1)
                if mask.numel() > 0:
                    tmp.index_fill_(0, mask, 0)

            targets = tmp

        return self.criterion(log_inputs, targets)


class SigmoidEntropy():
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, logits):
        p1 = torch.sigmoid(logits)
        p2 = 1 - p1
        entropy = -(p1 * torch.log(torch.clamp(p1, 1e-6)) + p2 * torch.log(torch.clamp(p2, 1e-6)))

        if self.reduction == "mean":
            return entropy.mean()
        if self.reduction == "sum":
            return entropy.sum()
        return entropy


class SigmoidKLDivLoss(nn.KLDivLoss):
    def forward(self, logits, targets):
        proba = torch.sigmoid(logits)
        log_prob = torch.log(torch.clamp(torch.stack([proba, 1 - proba], dim=1), 1e-6))
        targets = torch.stack([targets, 1 - targets], dim=1)

        return super().forward(log_prob, targets)

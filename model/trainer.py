import os

import math
import numpy as np
import torch
import torch.nn as nn
from apex import amp
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .optim import AdamW
from .losses import SigmoidEntropy, SigmoidKLDivLoss, LabelSmoothingLoss


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, value):
        self._sum += value
        self._count += 1

    def __call__(self):
        if self._count:
            return self._sum / self._count
        return 0


class AucMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._predictions = None
        self._targets = None

    def update(self, predictions, targets):
        if self._predictions is None and self._targets is None:
            self._predictions = predictions
            self._targets = targets
        else:
            assert self._predictions is not None and self._targets is not None
            self._predictions = np.concatenate([self._predictions, predictions])
            self._targets = np.concatenate([self._targets, targets])

    def __call__(self):
        if self._predictions is None and self._targets is None:
            return 0
        return roc_auc_score(self._targets, self._predictions)


class LenMatchBatchSampler(BatchSampler):
    def __iter__(self):
        buckets = {}
        for idx in self.sampler:
            length = len(self.sampler.data_source[idx][0])
            bucket_id = length // 64

            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(idx)

            if len(buckets[bucket_id]) == self.batch_size:
                yield buckets[bucket_id]
                buckets[bucket_id] = []

        leftover = [idx for bucket in buckets.values() for idx in bucket]
        batch = []
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class Trainer:
    def __init__(self, model, optimizer_params={}, loss_params={}, amp_params={}, checkpoint_dir=None, device=None, n_jobs=0):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        smoothing = loss_params.get('smoothing', 0)
        lm_weight = loss_params.get('lm_weight', 1)
        cls_weight = loss_params.get('cls_weight', 1)
        entropy_weight = loss_params.get('entropy_weight', 0)
        lr = optimizer_params.get('lr', 1e-3)
        lr_decay = optimizer_params.get('lr_decay', 0)
        weight_decay = optimizer_params.get('weight_decay', 0)
        amsgrad = optimizer_params.get('amsgrad', False)
        warmap = optimizer_params.get('warmap', 2000)
        opt_level = amp_params.get('opt_level', 'O0')
        loss_scale = amp_params.get('loss_scale', None)

        self.model = model.to(device)
        self.entropy = SigmoidEntropy()
        self.criterion = SigmoidKLDivLoss(reduction='batchmean').to(device)
        self.lm_criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings,
                                               ignore_index=self.model.padding_idx,
                                               smoothing=smoothing).to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias']
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=lr,
                               weight_decay=weight_decay,
                               amsgrad=amsgrad)

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level, loss_scale=loss_scale)

        def scheduler_func(iteration):
            if iteration <= warmap:
                return iteration / warmap
            return 1 - lr_decay * iteration

        self.scheduler = LambdaLR(self.optimizer, scheduler_func)

        self.lm_weight = lm_weight
        self.cls_weight = cls_weight
        self.entropy_weight = entropy_weight
        self.last_epoch = 0
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.n_jobs = n_jobs

    def _train_epoch(self, train_dataloader, batch_split):
        tqdm_train_dataloader = tqdm(train_dataloader, desc=f'Train, epoch #{self.last_epoch}')
        self.model.train()
        self.optimizer.zero_grad()

        entropy, cls_loss, lm_loss = AvgMeter(), AvgMeter(), AvgMeter()
        for i, (tokens, labels) in enumerate(tqdm_train_dataloader, 1):
            tokens, labels = tokens.to(self.device), labels.to(self.device)
            cls_logits, lm_logits = self.model(tokens)

            batch_entropy = self.entropy(cls_logits)
            batch_cls_loss = self.criterion(cls_logits, labels)
            batch_lm_loss = self.lm_criterion(lm_logits[:, :-1].contiguous().view(-1, lm_logits.shape[-1]), tokens[:, 1:].contiguous().view(-1))
            full_loss = (self.cls_weight * batch_cls_loss + self.lm_weight * batch_lm_loss - self.entropy_weight * batch_entropy) / batch_split

            with amp.scale_loss(full_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            if i % batch_split == 0:
                self.scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

            entropy.update(batch_entropy.item())
            cls_loss.update(batch_cls_loss.item())
            lm_loss.update(batch_lm_loss.item())
            tqdm_train_dataloader.set_postfix({'cls_loss': cls_loss(), 'lm_loss': lm_loss(), 'entropy': entropy()})

    def _test_epoch(self, test_dataloader):
        with torch.no_grad():
            tqdm_test_dataloader = tqdm(test_dataloader, desc=f'Test, epoch #{self.last_epoch}')
            self.model.eval()

            entropy, cls_loss, lm_loss, auc = AvgMeter(), AvgMeter(), AvgMeter(), AucMeter()
            for tokens, labels in tqdm_test_dataloader:
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                cls_logits, lm_logits = self.model(tokens)

                batch_entropy = self.entropy(cls_logits)
                batch_cls_loss = self.criterion(cls_logits, labels)
                batch_lm_loss = self.lm_criterion(lm_logits[:, :-1].contiguous().view(-1, lm_logits.shape[-1]), tokens[:, 1:].contiguous().view(-1))
                predictions = self.model.predict_from_logits(cls_logits)

                entropy.update(batch_entropy.item())
                cls_loss.update(batch_cls_loss.item())
                lm_loss.update(batch_lm_loss.item())
                auc.update(predictions.cpu().numpy(), (labels >= 0.5).cpu().numpy())
                tqdm_test_dataloader.set_postfix({'cls_loss': cls_loss(), 'lm_loss': lm_loss(), 'entropy': entropy(), 'auc': auc()})

            result_metric = auc()

            return result_metric

    def _save_checkpoint(self, name):
        if self.checkpoint_dir is None:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def _collate_func(self, data):
        tokens, labels = zip(*data)
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.model.padding_idx)
        labels = torch.stack(labels, dim=0)
        
        return tokens, labels

    def train(self, train_data, n_epochs, train_batch_size, train_batch_split=1, test_data=None, test_batch_size=None,
              save_last=False, save_best=False):
        sampler = RandomSampler(train_data)
        batch_sampler = LenMatchBatchSampler(sampler, batch_size=(train_batch_size + train_batch_split - 1) // train_batch_split, drop_last=False)
        train_dataloader = DataLoader(train_data, batch_sampler=batch_sampler, collate_fn=self._collate_func, num_workers=self.n_jobs)

        if test_data is not None:
            if test_batch_size is None:
                test_batch_size = train_batch_size
            test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False,
                                         collate_fn=self._collate_func, num_workers=self.n_jobs)

        best_metric = float("-inf")
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self._train_epoch(train_dataloader, train_batch_split)

            if save_last:
                self._save_checkpoint("last_checkpoint.pt")

            if test_data is not None:
                torch.cuda.empty_cache()
                metric = self._test_epoch(test_dataloader)

                if save_best:
                    if metric > best_metric:
                        best_metric = metric
                        self._save_checkpoint("best_checkpoint.pt")

            self.last_epoch += 1

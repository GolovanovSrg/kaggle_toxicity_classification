import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from model.gpt_utils import MODEL_INFO, prepare_gpt_weights, prepare_bpe_vocab, prepare_bpe_codes, load_gpt_weights
from model.text import get_vocab
from model.model import ClassificationModel, Predictor
from model.trainer import Trainer
from model.datasets import TextClassificationDataset


def set_seed(seed=0):
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def get_model_vocab(config):
    model_type = config['model_type']
    dropout = config['dropout']
    sparse_embedding = config['sparse_embedding']
    constant_pos_embedding = config['constant_pos_embedding']
    n_checkpoint_segments = config['n_checkpoint_segments']
    vocab_dir = config['vocab_dir']
    parameters_dir = config['parameters_dir']
    checkpoint_path = config['checkpoint_path']

    tokenizer_type = 'gpt2' if model_type.startswith('gpt2') else 'gpt'
    vocab_path = os.path.join(vocab_dir, tokenizer_type + '_bpe.vocab')
    codes_path = os.path.join(vocab_dir, tokenizer_type + '_bpe.codes')
    prepare_bpe_vocab(vocab_path, model_type)
    prepare_bpe_codes(codes_path, model_type)
    vocab = get_vocab(vocab_path, codes_path, tokenizer_type)

    model_config = MODEL_INFO[model_type]['config']
    model = ClassificationModel(n_layers=model_config['n_layers'],
                                n_embeddings=len(vocab),
                                n_pos_embeddings=model_config['n_pos_embeddings'],
                                embedding_dim=model_config['embeddings_size'],
                                n_heads=model_config['n_heads'],
                                normalize_before=model_config['normalize_before'],
                                padding_idx=vocab.pad_id,
                                dropout=dropout,
                                embedding_dropout=dropout,
                                attn_dropout=dropout,
                                ff_dropout=dropout,
                                sparse_embedding=sparse_embedding,
                                constant_pos_embedding=constant_pos_embedding,
                                n_checkpoint_segments=n_checkpoint_segments,
                                future_mask=True)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f'Checkpoint from {checkpoint_path}')
    elif parameters_dir is not None:
        parameters_path = os.path.join(parameters_dir, model_type + '_parameters.pt')
        prepare_gpt_weights(parameters_path, model_type)
        parameters = torch.load(parameters_path, map_location='cpu')
        load_gpt_weights(model.encoder, parameters, vocab.n_special_tokens)

    return model, vocab


def get_predictor(config):
    if config['checkpoint_path'] is None:
        raise ValueError('Checkpoint path is not set')

    model, vocab = get_model_vocab(config)
    device = config['device']
    predictor = Predictor(model, vocab, device)
    return predictor


def get_trainer(config, model):
    optimizer_params = {'lr': config['lr'],
                        'lr_decay': config['lr_decay'],
                        'weight_decay': config['weight_decay'],
                        'amsgrad': config['amsgrad']}
    loss_params = {'smoothing': config['smoothing'],
                   'lm_weight': config['lm_weight'],
                   'cls_weight': config['cls_weight']}
    amp_params = {'opt_level': config['opt_level'],
                  'loss_scale': config['loss_scale']}
    checkpoint_dir = config['checkpoint_dir']
    device = config['device']
    n_jobs = config['n_jobs']

    trainer = Trainer(model, optimizer_params, loss_params, amp_params, checkpoint_dir, device, n_jobs)

    return trainer


def train_val_split_dataset(data_path, train_path, val_path, validation_size=0.2, seed=0):
    data = pd.read_csv(data_path, index_col='id')
    labels = data['target'].values

    train_data, val_data, _, _ = train_test_split(data, labels,
                                                  test_size=validation_size,
                                                  random_state=seed,
                                                  shuffle=True,
                                                  stratify=(labels >= 0.5))

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    train_data.to_csv(train_path)
    val_data.to_csv(val_path)


def get_train_dataset(data_path, vocabulary, max_positions=1024, neg_sample=-1):
    data = pd.read_csv(data_path)
    texts = data['comment_text'].values
    labels = data['target'].values

    if neg_sample >= 0:
        is_pos = labels >= 0.5
        is_neg = ~is_pos
        idxs = np.arange(len(labels))
        sample_idxs = np.random.choice(idxs[is_neg], neg_sample * is_pos.sum())
        new_idxs = np.concatenate([idxs[is_pos], sample_idxs])
        texts, labels = texts[new_idxs], labels[new_idxs]

    return TextClassificationDataset(texts, labels, vocabulary, max_positions)


def get_test_dataset(data_path, vocabulary, max_positions=1024):
    data = pd.read_csv(data_path)
    texts = data['comment_text'].values
    return TextClassificationDataset(texts, None, vocabulary, max_positions)

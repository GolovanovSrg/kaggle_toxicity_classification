import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .modules import Transformer


class ClassificationModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx, n_heads,
                 dropout=0, embedding_dropout=0, attn_dropout=0, ff_dropout=0, sparse_embedding=False,
                 constant_pos_embedding=False, normalize_before=False, future_mask=True, n_checkpoint_segments=None):
        super().__init__()

        self.n_embeddings = n_embeddings
        self.padding_idx = padding_idx
        self.n_pos_embeddings = n_pos_embeddings
        self.encoder = Transformer(n_layers=n_layers,
                                   n_embeddings=n_embeddings,
                                   n_pos_embeddings=n_pos_embeddings,
                                   embedding_dim=embedding_dim, 
                                   padding_idx=padding_idx,
                                   n_heads=n_heads,
                                   dropout=dropout,
                                   embedding_dropout=embedding_dropout,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   sparse_embedding=sparse_embedding,
                                   normalize_before=normalize_before,
                                   constant_pos_embedding=constant_pos_embedding,
                                   n_checkpoint_segments=n_checkpoint_segments,
                                   future_mask=future_mask)
        self.cls_layer = nn.Linear(embedding_dim, 1)
        self.lm_layer = nn.Linear(embedding_dim, n_embeddings, bias=False)
        self.lm_layer.weight = self.encoder.embedding.tok_embedding.weight

    def predict_from_logits(self, logits):
        return torch.sigmoid(logits).view(-1)

    def predict(self, x):
        logits, _ = self.forward(x)
        return self.predict_from_logits(logits)

    def forward(self, x):
        x, padding_mask = self.encoder(x)
        lengths = (~padding_mask).long().sum(dim=-1)
        lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        cls_output = self.cls_layer(x.gather(1, lengths-1))
        cls_output = cls_output.squeeze(-1).squeeze(-1)
        lm_output = self.lm_layer(x)
        return cls_output, lm_output


class Predictor:
    def __init__(self, model, vocab, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        self._model = model.to(device)
        self._vocab = vocab
        self._device = device

    def __call__(self, texts):
        self._model.eval()
        with torch.no_grad():
            tokens = [torch.tensor(self._vocab.string2ids(text)[:self._model.n_pos_embeddings-1]+[self._vocab.eos_id], dtype=torch.long) for text in texts]
            tokens = pad_sequence(tokens, batch_first=True, padding_value=self._model.padding_idx)

            tokens = tokens.to(self._device)
            predictions = self._model.predict(tokens)

            return predictions.cpu().numpy()

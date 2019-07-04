import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_positions):
        super().__init__()

        self.vocab = vocab
        self.max_positions = max_positions
        self.data = list(zip(texts, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = torch.tensor(self.vocab.string2ids(text)[:self.max_positions-1]+[self.vocab.eos_id], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)

        return tokens, label

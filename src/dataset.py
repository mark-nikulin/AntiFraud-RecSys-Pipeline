"""
dataset.py — PyTorch Dataset с Negative Sampling для Anti-Fraud RecSys.

Ключевая идея:
  - В методе __getitem__ к каждому позитивному примеру (реальная следующая
    транзакция) добавляется k случайных негативных сэмплов из пула всех целевых
    транзакций в датасете.
  - Это позволяет обучить модель на BPR Loss: score(pos) > score(neg).
  - В режиме is_train=False (val/test) негативные сэмплы не нужны —
    возвращаем только позитив + метку для вычисления ROC-AUC.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def load_processed_data(config) -> tuple[dict, dict, dict, dict]:
    """
    Загружает train/val/test numpy-массивы и metadata.json из processed_dir.

    Returns:
        train_data, val_data, test_data, metadata
    """
    p = config.processed_dir
    meta_path = p / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Файл {meta_path} не найден.\n"
            "Сначала запусти: python src/prepare_data.py"
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    keys = [
        "seq_amt", "seq_product", "seq_card4", "seq_card6",
        "static",
        "tgt_amt", "tgt_product", "tgt_card4", "tgt_card6",
        "label",
    ]

    def _load(split: str) -> dict:
        return {k: np.load(p / f"{split}_{k}.npy") for k in keys}

    return _load("train"), _load("val"), _load("test"), metadata


class FraudDataset(Dataset):
    """
    Dataset для Anti-Fraud модели.

    Каждый элемент содержит:
      - seq:        история из seq_len транзакций (4 массива)
      - static:     статический профиль пользователя (6 float)
      - pos_target: реальная следующая транзакция (позитив)
      - neg_target: случайная транзакция из пула (негатив) — только при is_train
      - label:      isFraud для позитивного примера (для eval)
    """

    def __init__(self, data: dict, config, is_train: bool = True):
        self.is_train = is_train
        self.neg_k    = config.neg_samples
        self.rng      = np.random.default_rng(config.seed)

        # История (sequential)
        self.seq_amt     = torch.FloatTensor(data["seq_amt"])       # [N, seq_len]
        self.seq_product = torch.LongTensor(data["seq_product"])    # [N, seq_len]
        self.seq_card4   = torch.LongTensor(data["seq_card4"])      # [N, seq_len]
        self.seq_card6   = torch.LongTensor(data["seq_card6"])      # [N, seq_len]

        # Профиль пользователя (static)
        self.static = torch.FloatTensor(data["static"])             # [N, 6]

        # Целевые транзакции (позитивы)
        self.tgt_amt     = torch.FloatTensor(data["tgt_amt"])       # [N]
        self.tgt_product = torch.LongTensor(data["tgt_product"])    # [N]
        self.tgt_card4   = torch.LongTensor(data["tgt_card4"])      # [N]
        self.tgt_card6   = torch.LongTensor(data["tgt_card6"])      # [N]

        # Метки (для evaluate)
        self.labels = torch.FloatTensor(data["label"].astype(np.float32))  # [N]

        self.N = len(self.labels)

        # Пул всех транзакций — источник негативных сэмплов.
        # Индексируем весь датасет целиком: случайный индекс ≠ текущий ≈ "не пользователь"
        self._all_indices = np.arange(self.N)

    def __len__(self) -> int:
        return self.N

    def _sample_negative(self, pos_idx: int) -> int:
        """Случайно выбираем транзакцию из пула, избегая позитивного индекса."""
        while True:
            neg_idx = int(self.rng.integers(0, self.N))
            if neg_idx != pos_idx:
                return neg_idx

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "seq": {
                "amt":     self.seq_amt[idx],      # [seq_len]
                "product": self.seq_product[idx],  # [seq_len]
                "card4":   self.seq_card4[idx],    # [seq_len]
                "card6":   self.seq_card6[idx],    # [seq_len]
            },
            "static": self.static[idx],            # [6]
            "pos_target": {
                "amt":     self.tgt_amt[idx],      # scalar
                "product": self.tgt_product[idx],  # scalar
                "card4":   self.tgt_card4[idx],    # scalar
                "card6":   self.tgt_card6[idx],    # scalar
            },
            "label": self.labels[idx],             # scalar
        }

        if self.is_train:
            neg_idx = self._sample_negative(idx)
            sample["neg_target"] = {
                "amt":     self.tgt_amt[neg_idx],
                "product": self.tgt_product[neg_idx],
                "card4":   self.tgt_card4[neg_idx],
                "card6":   self.tgt_card6[neg_idx],
            }

        return sample

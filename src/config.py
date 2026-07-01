"""
config.py — все гиперпараметры проекта в одном месте.
Используется как единый источник правды для prepare_data, train, evaluate.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Пути ────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    checkpoint_dir: Path = Path("checkpoints")
    logs_dir: Path = Path("runs")

    # ── Данные ──────────────────────────────────────────────────────────────
    seq_len: int = 10          # длина истории транзакций (sliding window)
    train_frac: float = 0.70   # доля обучающей выборки (time-based split)
    val_frac: float = 0.15     # доля валидационной выборки

    # ── Модель ──────────────────────────────────────────────────────────────
    emb_dim: int = 64              # размерность всех эмбеддингов
    num_heads: int = 4             # головы внимания в TransformerEncoder
    num_transformer_layers: int = 2
    dropout: float = 0.1
    static_dim: int = 6            # размер вектора статических фич пользователя
    static_hidden_dim: int = 128   # скрытый слой Static Encoder

    # ── Обучение ────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 30
    neg_samples: int = 4           # негативных сэмплов на 1 позитив
    grad_clip: float = 1.0
    early_stopping_patience: int = 5

    # ── Оценка ──────────────────────────────────────────────────────────────
    k_percent: float = 5.0         # Recall@K% — блокируем верхние K% по скору

    # ── Воспроизводимость ───────────────────────────────────────────────────
    seed: int = 42

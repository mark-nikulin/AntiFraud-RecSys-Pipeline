"""
model.py — Two-Tower архитектура AntiFraudModel.

Архитектура:
  ┌─────────────────────────────────────────────────────────────────┐
  │                        AntiFraudModel                          │
  │                                                                 │
  │  TransactionEmbedder          (общий для seq и target)          │
  │   ├── amt_proj   Linear(1 → emb_dim)                           │
  │   ├── product_emb Embedding(V_prod, emb_dim)                   │
  │   ├── card4_emb   Embedding(V_c4, emb_dim)                     │
  │   └── card6_emb   Embedding(V_c6, emb_dim)                     │
  │                                                                 │
  │  SequentialEncoder  (SASRec-упрощение)                         │
  │   ├── pos_emb   Embedding(seq_len, emb_dim)  learnable PE      │
  │   └── transformer  TransformerEncoder(L layers)                │
  │       → берём hidden state последнего токена H ∈ ℝ^emb_dim     │
  │                                                                 │
  │  StaticEncoder  (MLP)                                          │
  │   └── Linear → ReLU → Dropout → Linear → LayerNorm            │
  │       P ∈ ℝ^emb_dim                                            │
  │                                                                 │
  │  FusionHead                                                     │
  │   ├── cat([H, P]) → Linear(2*emb_dim → emb_dim) → ReLU        │
  │   └── user_repr ∈ ℝ^emb_dim                                    │
  │                                                                 │
  │  Scoring: score = dot(user_repr, target_emb)  ∈ ℝ              │
  └─────────────────────────────────────────────────────────────────┘

Метод forward() возвращает (pos_score, neg_score) для BPR Loss.
Метод predict() возвращает sigmoid(score) — вероятность фрода.
"""

import torch
import torch.nn as nn


class TransactionEmbedder(nn.Module):
    """Кодирует одну транзакцию (или батч) в единый вектор emb_dim."""

    def __init__(self, vocab_sizes: dict, emb_dim: int, dropout: float):
        super().__init__()
        # Категориальные фичи
        self.product_emb = nn.Embedding(vocab_sizes["ProductCD"], emb_dim, padding_idx=0)
        self.card4_emb   = nn.Embedding(vocab_sizes["card4"],     emb_dim, padding_idx=0)
        self.card6_emb   = nn.Embedding(vocab_sizes["card6"],     emb_dim, padding_idx=0)
        # Непрерывная фича (сумма транзакции)
        self.amt_proj = nn.Linear(1, emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(emb_dim)

    def forward(
        self,
        amt: torch.Tensor,      # [...] float
        product: torch.Tensor,  # [...] long
        card4: torch.Tensor,    # [...] long
        card6: torch.Tensor,    # [...] long
    ) -> torch.Tensor:
        """Возвращает тензор формы [..., emb_dim]."""
        x = (
            self.amt_proj(amt.unsqueeze(-1))   # [..., emb_dim]
            + self.product_emb(product)
            + self.card4_emb(card4)
            + self.card6_emb(card6)
        )
        return self.dropout(self.norm(x))


class SequentialEncoder(nn.Module):
    """
    Упрощённый SASRec: TransformerEncoder с обучаемым позиционным кодированием.
    Принимает последовательность из seq_len векторов, возвращает вектор
    последней позиции (самой свежей транзакции).
    """

    def __init__(self, emb_dim: int, num_heads: int, num_layers: int,
                 seq_len: int, dropout: float):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, emb_dim)  # learnable positional
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: стабильнее при обучении
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, emb_dim]
        Returns: [B, emb_dim]  — hidden state последней позиции
        """
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]
        x = x + self.pos_emb(positions)                             # [B, L, emb_dim]
        x = self.encoder(x)                                         # [B, L, emb_dim]
        return x[:, -1, :]                                          # [B, emb_dim]


class StaticEncoder(nn.Module):
    """MLP для статического профиля пользователя: [mean_amt, std_amt, ...]."""

    def __init__(self, static_dim: int, hidden_dim: int, emb_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, static_dim] → [B, emb_dim]"""
        return self.net(x)


class AntiFraudModel(nn.Module):
    """
    Гибридная Two-Tower модель для детекции фрода.

    Параметры:
        config     — объект Config с гиперпараметрами
        vocab_sizes — словарь с размерами словарей для Embedding слоёв,
                      например: {"ProductCD": 6, "card4": 5, "card6": 4}
    """

    def __init__(self, config, vocab_sizes: dict):
        super().__init__()
        emb_dim = config.emb_dim

        # Общий кодировщик транзакций (для истории И для целевых транзакций)
        self.tx_embedder = TransactionEmbedder(vocab_sizes, emb_dim, config.dropout)

        # Башня 1: Sequential (история)
        self.seq_encoder = SequentialEncoder(
            emb_dim=emb_dim,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers,
            seq_len=config.seq_len,
            dropout=config.dropout,
        )

        # Башня 2: Static (профиль пользователя)
        self.static_encoder = StaticEncoder(
            static_dim=config.static_dim,
            hidden_dim=config.static_hidden_dim,
            emb_dim=emb_dim,
            dropout=config.dropout,
        )

        # Fusion: объединяем H и P → user_repr
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def _embed_seq(self, seq: dict) -> torch.Tensor:
        """Кодирует историю транзакций в [B, seq_len, emb_dim]."""
        return self.tx_embedder(
            seq["amt"], seq["product"], seq["card4"], seq["card6"]
        )

    def _embed_target(self, target: dict) -> torch.Tensor:
        """Кодирует одну транзакцию в [B, emb_dim]."""
        return self.tx_embedder(
            target["amt"], target["product"], target["card4"], target["card6"]
        )

    def encode_user(self, seq: dict, static: torch.Tensor) -> torch.Tensor:
        """
        Получаем итоговое представление пользователя.
        seq:    словарь батчей [B, seq_len]
        static: [B, static_dim]
        Returns: user_repr [B, emb_dim]
        """
        seq_emb    = self._embed_seq(seq)           # [B, seq_len, emb_dim]
        H          = self.seq_encoder(seq_emb)      # [B, emb_dim]
        P          = self.static_encoder(static)    # [B, emb_dim]
        combined   = torch.cat([H, P], dim=-1)      # [B, 2*emb_dim]
        return self.fusion(combined)                # [B, emb_dim]

    def score(self, user_repr: torch.Tensor, target: dict) -> torch.Tensor:
        """
        Dot-product скор: насколько транзакция target характерна для user_repr.
        Returns: [B] scalar scores
        """
        target_emb = self._embed_target(target)     # [B, emb_dim]
        return (user_repr * target_emb).sum(dim=-1) # [B]

    # ── Основные методы ──────────────────────────────────────────────────────

    def forward(
        self,
        seq: dict,
        static: torch.Tensor,
        pos_target: dict,
        neg_target: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Используется во время обучения (BPR Loss).
        Returns: (pos_score [B], neg_score [B])
        """
        user_repr = self.encode_user(seq, static)
        pos_score = self.score(user_repr, pos_target)
        neg_score = self.score(user_repr, neg_target)
        return pos_score, neg_score

    @torch.no_grad()
    def predict(
        self,
        seq: dict,
        static: torch.Tensor,
        target: dict,
    ) -> torch.Tensor:
        """
        Используется во время инференса/оценки.
        Returns: fraud probability [B] ∈ (0, 1)
        """
        user_repr = self.encode_user(seq, static)
        score     = self.score(user_repr, target)
        return torch.sigmoid(score)

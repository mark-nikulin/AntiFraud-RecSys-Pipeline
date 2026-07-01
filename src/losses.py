"""
losses.py — функции потерь для обучения RecSys модели на задаче ранжирования.

BPR Loss (Bayesian Personalized Ranking):
  Суть: score(позитив) должен быть ВЫШЕ, чем score(негатив).
  Формула: Loss = -log(σ(score_pos - score_neg))
  Это дословно то, что требуется в вакансии: «loss-функции для задач ранжирования».

Contrastive Loss (альтернатива с margin):
  Loss = max(0, margin - score_pos + score_neg)
"""

import torch
import torch.nn.functional as F


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking Loss.

    Args:
        pos_score: [B] — скоры реальных (позитивных) транзакций
        neg_score:  [B] — скоры случайных (негативных) транзакций

    Returns:
        scalar — среднее значение потерь по батчу
    """
    return -F.logsigmoid(pos_score - neg_score).mean()


def contrastive_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Contrastive Loss с margin.

    Args:
        pos_score: [B]
        neg_score: [B]
        margin:    минимальный требуемый зазор между скорами

    Returns:
        scalar
    """
    return torch.clamp(margin - pos_score + neg_score, min=0.0).mean()

"""
evaluate.py — метрики для оценки Anti-Fraud модели.

Почему не Accuracy?
  При 3.5% фрода модель, всегда предсказывающая «не фрод», даёт 96.5% accuracy.
  Поэтому используем метрики, специфичные для несбалансированных классов:

  ROC-AUC  — глобальная способность ранжировать фрод выше нормы.
             Цель: > 0.85
  PR-AUC   — Area Under Precision-Recall кривой, более информативна при
             сильном дисбалансе. Цель: > 0.50
  Recall@K% — какую долю фрода ловим, блокируя верхние K% по скору.
             Аналог «накрытия» в продакшн-системах. Цель: > 0.40 при K=5%
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k_percent: float) -> float:
    """
    Recall when flagging the top-k% most suspicious transactions.

    Бизнес-интерпретация: «Сколько процентов мошенничества мы поймаем,
    если будем блокировать/проверять верхние K% всех транзакций?»

    Args:
        scores:    массив вероятностей/скоров [N]
        labels:    бинарные метки [N]
        k_percent: верхний процент (например, 5.0 = топ-5%)

    Returns:
        float ∈ [0, 1]
    """
    n = len(scores)
    k = max(1, int(round(n * k_percent / 100)))

    top_k_idx    = np.argpartition(scores, -k)[-k:]
    total_fraud  = labels.sum()

    if total_fraud == 0:
        return 0.0

    fraud_caught = labels[top_k_idx].sum()
    return float(fraud_caught / total_fraud)


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    k_percent: float = 5.0,
) -> dict[str, float]:
    """
    Считает ROC-AUC, PR-AUC и Recall@K%.

    Args:
        scores:    предсказанные вероятности/скоры [N]
        labels:    бинарные метки (0 = нормально, 1 = фрод) [N]
        k_percent: порог для Recall@K%

    Returns:
        dict с ключами: roc_auc, pr_auc, recall_at_k
    """
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if labels.sum() == 0:
        # Нет позитивов в батче — метрики не определены
        return {"roc_auc": 0.0, "pr_auc": 0.0, "recall_at_k": 0.0}

    return {
        "roc_auc":    float(roc_auc_score(labels, scores)),
        "pr_auc":     float(average_precision_score(labels, scores)),
        "recall_at_k": recall_at_k(scores, labels, k_percent),
    }

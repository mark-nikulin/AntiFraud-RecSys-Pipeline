# AntiFraud-RecSys-Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch)
![TensorBoard](https://img.shields.io/badge/TensorBoard-logging-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end Anti-Fraud Recommender System с Two-Tower архитектурой, Negative Sampling и BPR Loss — построен на PyTorch **без** высокоуровневых оберток.

---

## Бизнес-задача

В банке — миллионы транзакций в день. Большинство нормальные. Нужно **в режиме реального времени** оценивать вероятность мошенничества, опираясь на историю пользователя и его профиль.

ML-задача: ранжировать транзакции по степени подозрительности так, чтобы при блокировке **5% самых подозрительных** поймать **максимум фрода** с минимумом ложных срабатываний.

---

## Архитектура модели

```
                    ┌────────────────────────────────────────┐
                    │          AntiFraudModel                │
                    │                                        │
  История (N tx) ──►│  TransactionEmbedder                  │
                    │   amt  → Linear(1→D)    ┐             │
                    │   prod → Embedding(V,D)  ├─ sum → D   │
                    │   card → Embedding(V,D) ┘             │
                    │              │                         │
                    │   SequentialEncoder (SASRec)           │
                    │   Learnable PE + TransformerEncoder    │
                    │   → H ∈ ℝᴰ  (последний токен)         │
                    │                                        │
  Профиль ─────────►│  StaticEncoder (MLP)                  │
  [mean_amt,        │   Linear→ReLU→Linear→LayerNorm        │
   std_amt, ...]    │   → P ∈ ℝᴰ                            │
                    │                                        │
                    │  FusionHead                            │
                    │   cat([H, P]) → Linear(2D→D) → ReLU   │
                    │   → user_repr ∈ ℝᴰ                     │
                    │                                        │
  Target tx ───────►│  dot(user_repr, target_emb) → score   │
                    │  sigmoid(score) → P(fraud) ∈ (0,1)    │
                    └────────────────────────────────────────┘
```

### Ключевые технические решения

| Компонент | Решение | Зачем |
|---|---|---|
| Sequential Encoder | TransformerEncoder (Pre-LN) | Улавливает временные паттерны поведения |
| Positional Encoding | Learnable Embedding | Адаптируется к распределению данных |
| Negative Sampling | Uniform sampling из пула транзакций | Учим модель отличать аномальные паттерны |
| Loss | BPR Loss: `-log σ(score_pos − score_neg)` | Прямая оптимизация ранжирования |
| Train/Test split | Time-based (не random) | Предотвращает data leakage из будущего |
| Early stopping | По PR-AUC на валидации | PR-AUC — правильная метрика при дисбалансе |

---

## Данные: IEEE-CIS Fraud Detection

- **Источник**: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- ~590K транзакций | ~3.5% фрода
- Группировка по `card1` (proxy user_id) | Сортировка по `TransactionDT`

### Sequential фичи (per transaction)
- `TransactionAmt` → log1p нормализация → `Linear(1, D)`
- `ProductCD`, `card4`, `card6` → LabelEncoding → `Embedding(V, D)`

### Static фичи (user profile)
- `mean_amt`, `std_amt` — статистика истории
- `card4`, `card6` — тип карты
- `addr1` — регион
- `log_tx_count` — активность пользователя

---

## Быстрый старт

```bash
# 1. Клонируем и устанавливаем зависимости
git clone https://github.com/<your-username>/AntiFraud-RecSys-Pipeline
cd AntiFraud-RecSys-Pipeline
pip install -r requirements.txt

# 2. Скачиваем датасет (нужен Kaggle API token)
#    Подробнее: https://www.kaggle.com/docs/api
bash scripts/download_data.sh

# 3. Препроцессинг
python src/prepare_data.py

# 4. Обучение
python src/train.py

# 5. Мониторинг (в отдельном терминале)
tensorboard --logdir runs/
```

---

## Результаты

| Метрика | Значение | Цель |
|---|---|---|
| ROC-AUC | — | > 0.85 |
| PR-AUC | — | > 0.50 |
| Recall@5% | — | > 0.40 |

> Метрики будут заполнены после обучения на полном датасете.

---

## Структура репозитория

```
AntiFraud-RecSys-Pipeline/
├── src/
│   ├── config.py        # Все гиперпараметры в одном месте
│   ├── prepare_data.py  # Препроцессинг, sliding windows, time-based split
│   ├── dataset.py       # FraudDataset с Negative Sampling
│   ├── model.py         # AntiFraudModel (Two-Tower + SASRec)
│   ├── losses.py        # BPR Loss, Contrastive Loss
│   ├── train.py         # Ручной training loop + TensorBoard
│   └── evaluate.py      # ROC-AUC, PR-AUC, Recall@K%
├── scripts/
│   ├── download_data.sh # Скачивание датасета через Kaggle CLI
│   └── run_training.sh  # Запуск полного пайплайна
├── requirements.txt
└── README.md
```

---

## Стек

`Python 3.10+` · `PyTorch 2.2` · `pandas` · `numpy` · `scikit-learn` · `TensorBoard`
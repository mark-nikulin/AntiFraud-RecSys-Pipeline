"""
train.py — ручной цикл обучения Anti-Fraud модели с TensorBoard логированием.

Запуск:
  python src/train.py

Мониторинг:
  tensorboard --logdir runs/
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.dataset import FraudDataset, load_processed_data
from src.evaluate import compute_metrics
from src.losses import bpr_loss
from src.model import AntiFraudModel


# ── Воспроизводимость ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Утилиты для работы с батчем ─────────────────────────────────────────────

def to_device(batch: dict, device: torch.device) -> dict:
    """Рекурсивно переносит тензоры батча на устройство."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v.to(device)
    return out


# ── Один эпохи обучения ──────────────────────────────────────────────────────

def train_epoch(
    model: AntiFraudModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        pos_score, neg_score = model(
            seq=batch["seq"],
            static=batch["static"],
            pos_target=batch["pos_target"],
            neg_target=batch["neg_target"],
        )

        loss = bpr_loss(pos_score, neg_score)
        loss.backward()

        # Gradient clipping — важно для трансформеров
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Оценка на val/test ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model: AntiFraudModel,
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_scores = []
    all_labels = []

    for batch in loader:
        batch = to_device(batch, device)

        scores = model.predict(
            seq=batch["seq"],
            static=batch["static"],
            target=batch["pos_target"],
        )

        all_scores.append(scores.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    return compute_metrics(all_scores, all_labels, config.k_percent)


# ── Главная функция ──────────────────────────────────────────────────────────

def main() -> None:
    config = Config()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Данные ───────────────────────────────────────────────────────────────
    train_data, val_data, test_data, metadata = load_processed_data(config)

    train_ds = FraudDataset(train_data, config=config, is_train=True)
    val_ds   = FraudDataset(val_data,   config=config, is_train=False)
    test_ds  = FraudDataset(test_data,  config=config, is_train=False)

    num_workers = 4 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=num_workers,
    )

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ── Модель ───────────────────────────────────────────────────────────────
    vocab_sizes = metadata["vocab_sizes"]
    model = AntiFraudModel(config, vocab_sizes).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров модели: {n_params:,}")

    # ── Оптимизатор и scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.logs_dir))
    print(f"TensorBoard: tensorboard --logdir {config.logs_dir}")

    # ── Чекпоинты ────────────────────────────────────────────────────────────
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = config.checkpoint_dir / "best_model.pt"

    # ── Цикл обучения ────────────────────────────────────────────────────────
    best_pr_auc      = 0.0
    patience_counter = 0

    print("\n" + "=" * 65)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'ROC-AUC':>8} | {'PR-AUC':>7} | {'R@K%':>6} | LR")
    print("=" * 65)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config, device)
        val_metrics = eval_epoch(model, val_loader, config, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>6d} | {train_loss:>8.4f} | "
            f"{val_metrics['roc_auc']:>8.4f} | "
            f"{val_metrics['pr_auc']:>7.4f} | "
            f"{val_metrics['recall_at_k']:>6.4f} | {lr:.2e}"
        )

        # TensorBoard logging
        writer.add_scalar("Loss/train",        train_loss,                    epoch)
        writer.add_scalar("Val/roc_auc",        val_metrics["roc_auc"],       epoch)
        writer.add_scalar("Val/pr_auc",         val_metrics["pr_auc"],        epoch)
        writer.add_scalar("Val/recall_at_k",    val_metrics["recall_at_k"],   epoch)
        writer.add_scalar("LR",                 lr,                           epoch)

        # Early stopping по PR-AUC
        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "vocab_sizes": vocab_sizes,
                },
                best_ckpt,
            )
            print(f"  ✓ Checkpoint saved (PR-AUC={best_pr_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping after epoch {epoch}.")
                break

    # ── Финальная оценка на тесте ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST SET EVALUATION (best checkpoint)")
    print("=" * 65)

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = eval_epoch(model, test_loader, config, device)

    print(f"  ROC-AUC:      {test_metrics['roc_auc']:.4f}   (цель > 0.85)")
    print(f"  PR-AUC:       {test_metrics['pr_auc']:.4f}   (цель > 0.50)")
    print(f"  Recall@{config.k_percent}%:  {test_metrics['recall_at_k']:.4f}   (цель > 0.40)")

    writer.add_hparams(
        hparam_dict={
            "lr": config.lr,
            "emb_dim": config.emb_dim,
            "batch_size": config.batch_size,
            "neg_samples": config.neg_samples,
        },
        metric_dict={
            "test/roc_auc": test_metrics["roc_auc"],
            "test/pr_auc":  test_metrics["pr_auc"],
        },
    )
    writer.close()
    print(f"\n✓ Готово! Лучший чекпоинт: {best_ckpt}")


if __name__ == "__main__":
    main()

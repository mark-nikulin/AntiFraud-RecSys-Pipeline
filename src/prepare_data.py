"""
prepare_data.py — препроцессинг IEEE-CIS Fraud Detection датасета.

Что делает этот скрипт:
  1. Загружает train_transaction.csv + train_identity.csv, мёрджит их.
  2. Очищает и кодирует категориальные фичи (LabelEncoding, +1 для padding_idx=0).
  3. Нормализует TransactionAmt через log1p.
  4. Группирует транзакции по card1 (proxy user_id), сортирует по времени.
  5. Строит sliding windows длиной seq_len.
  6. Делает time-based train/val/test split.
  7. Сохраняет готовые numpy-массивы и metadata.json в data/processed/.

Запуск:
  python src/prepare_data.py
  python src/prepare_data.py --check   # только проверка размеров
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Добавляем корень проекта в sys.path, чтобы можно было запускать напрямую
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config


# ── Константы ────────────────────────────────────────────────────────────────
CAT_COLS = ["ProductCD", "card4", "card6"]   # категориальные поля → Embedding
USER_COL = "card1"                            # proxy для user_id
TIME_COL = "TransactionDT"
AMT_COL  = "TransactionAmt"
TARGET   = "isFraud"

# Статические фичи пользователя (6 штук — должно совпадать с config.static_dim)
# Порядок: [mean_amt, std_amt, card4_enc, card6_enc, addr1_norm, log_tx_count]
STATIC_DIM = 6


# ── Вспомогательные функции ──────────────────────────────────────────────────

def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Загружает и мёрджит transaction + identity файлы."""
    tx_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    if not tx_path.exists():
        raise FileNotFoundError(
            f"Файл {tx_path} не найден.\n"
            "Запусти: bash scripts/download_data.sh"
        )

    print("Загружаем train_transaction.csv...")
    tx = pd.read_csv(
        tx_path,
        usecols=["TransactionID", TARGET, TIME_COL, AMT_COL,
                 "ProductCD", "card1", "card4", "card6", "addr1"],
        dtype={TARGET: "int8"},
    )

    if id_path.exists():
        print("Загружаем train_identity.csv (не используется напрямую)...")
        # Зарезервировано для расширения — сейчас используем только транзакции
        pass

    print(f"  Строк загружено: {len(tx):,}")
    return tx


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encoding категориальных колонок, словари сохраняются в metadata."""
    encoders = {}
    vocab_sizes = {}

    for col in CAT_COLS:
        df[col] = df[col].fillna("__NA__").astype(str)
        le = LabelEncoder()
        # +1 чтобы оставить 0 для padding в nn.Embedding
        df[col + "_enc"] = le.fit_transform(df[col]) + 1
        encoders[col] = le
        vocab_sizes[col] = int(df[col + "_enc"].max()) + 1  # +1 для padding

    return df, vocab_sizes


def normalize_amount(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """log1p + стандартизация суммы транзакции."""
    df["amt_log"] = np.log1p(df[AMT_COL].fillna(0.0))
    amt_mean = float(df["amt_log"].mean())
    amt_std  = float(df["amt_log"].std())
    df["amt_norm"] = (df["amt_log"] - amt_mean) / (amt_std + 1e-8)
    return df, {"amt_mean": amt_mean, "amt_std": amt_std}


def build_windows(df: pd.DataFrame, seq_len: int) -> dict[str, np.ndarray]:
    """
    Для каждого пользователя (card1) строит sliding windows.

    Возвращает словарь numpy-массивов:
      seq_amt      [N, seq_len]  — история сумм
      seq_product  [N, seq_len]  — история ProductCD (int)
      seq_card4    [N, seq_len]  — история card4 (int)
      seq_card6    [N, seq_len]  — история card6 (int)
      static       [N, 6]        — статический профиль пользователя
      tgt_amt      [N]           — сумма целевой транзакции
      tgt_product  [N]           — ProductCD целевой транзакции
      tgt_card4    [N]           — card4 целевой транзакции
      tgt_card6    [N]           — card6 целевой транзакции
      label        [N]           — isFraud целевой транзакции
      tx_dt        [N]           — время целевой транзакции (для split)
    """
    records = {
        "seq_amt": [], "seq_product": [], "seq_card4": [], "seq_card6": [],
        "static": [],
        "tgt_amt": [], "tgt_product": [], "tgt_card4": [], "tgt_card6": [],
        "label": [], "tx_dt": [],
    }

    # Нормализация addr1 (billing region code, числовой)
    addr1_max = df["addr1"].dropna().max()
    if addr1_max == 0:
        addr1_max = 1.0

    groups = df.groupby(USER_COL, sort=False)
    n_skipped = 0

    for _, group in tqdm(groups, desc="Построение окон", unit="user"):
        group = group.sort_values(TIME_COL).reset_index(drop=True)
        n = len(group)

        # Нужно минимум seq_len+1 транзакций, чтобы получить хоть одно окно
        if n < seq_len + 1:
            n_skipped += 1
            continue

        # Векторизованное извлечение фич по группе
        amts     = group["amt_norm"].to_numpy(dtype=np.float32)
        products = group["ProductCD_enc"].to_numpy(dtype=np.int64)
        c4s      = group["card4_enc"].to_numpy(dtype=np.int64)
        c6s      = group["card6_enc"].to_numpy(dtype=np.int64)
        labels   = group[TARGET].to_numpy(dtype=np.int8)
        dts      = group[TIME_COL].to_numpy(dtype=np.float64)

        # Статические фичи (постоянные для пользователя)
        card4_val = int(c4s[0])
        card6_val = int(c6s[0])
        addr1_raw = group["addr1"].dropna()
        addr1_val = float(addr1_raw.mode()[0]) / addr1_max if len(addr1_raw) > 0 else 0.0
        log_count = float(np.log1p(n))

        for i in range(seq_len, n):
            hist_amts = amts[i - seq_len : i]
            mean_amt  = float(hist_amts.mean())
            std_amt   = float(hist_amts.std()) if seq_len > 1 else 0.0

            records["seq_amt"].append(hist_amts.copy())
            records["seq_product"].append(products[i - seq_len : i].copy())
            records["seq_card4"].append(c4s[i - seq_len : i].copy())
            records["seq_card6"].append(c6s[i - seq_len : i].copy())
            records["static"].append(
                np.array([mean_amt, std_amt, card4_val, card6_val, addr1_val, log_count],
                         dtype=np.float32)
            )
            records["tgt_amt"].append(amts[i])
            records["tgt_product"].append(products[i])
            records["tgt_card4"].append(c4s[i])
            records["tgt_card6"].append(c6s[i])
            records["label"].append(labels[i])
            records["tx_dt"].append(dts[i])

    print(f"  Пользователей пропущено (<{seq_len+1} транзакций): {n_skipped:,}")

    # Конвертируем в numpy
    out = {
        "seq_amt":     np.stack(records["seq_amt"]),             # [N, seq_len]
        "seq_product": np.stack(records["seq_product"]),         # [N, seq_len]
        "seq_card4":   np.stack(records["seq_card4"]),           # [N, seq_len]
        "seq_card6":   np.stack(records["seq_card6"]),           # [N, seq_len]
        "static":      np.stack(records["static"]),              # [N, 6]
        "tgt_amt":     np.array(records["tgt_amt"],     dtype=np.float32),
        "tgt_product": np.array(records["tgt_product"], dtype=np.int64),
        "tgt_card4":   np.array(records["tgt_card4"],   dtype=np.int64),
        "tgt_card6":   np.array(records["tgt_card6"],   dtype=np.int64),
        "label":       np.array(records["label"],       dtype=np.int8),
        "tx_dt":       np.array(records["tx_dt"],       dtype=np.float64),
    }
    return out


def time_based_split(
    data: dict[str, np.ndarray],
    train_frac: float,
    val_frac: float,
) -> tuple[dict, dict, dict]:
    """
    Разбиение по времени (не случайное!).
    Это критически важно для антифрода — утечка данных из будущего недопустима.
    """
    n = len(data["tx_dt"])
    # Сортируем по времени целевой транзакции
    order = np.argsort(data["tx_dt"])
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    def subset(idx):
        return {k: v[idx] for k, v in data.items()}

    train_idx = order[:n_train]
    val_idx   = order[n_train : n_train + n_val]
    test_idx  = order[n_train + n_val :]

    return subset(train_idx), subset(val_idx), subset(test_idx)


def save_split(data: dict[str, np.ndarray], out_dir: Path, split: str) -> None:
    """Сохраняем каждый массив как отдельный .npy файл."""
    for key, arr in data.items():
        np.save(out_dir / f"{split}_{key}.npy", arr)
    print(f"  [{split}] {len(data['label']):,} окон "
          f"(фрод: {data['label'].sum():,}, "
          f"{data['label'].mean() * 100:.2f}%)")


def main(check_only: bool = False) -> None:
    config = Config()
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Загрузка ──────────────────────────────────────────────────────────
    df = load_raw_data(config.raw_dir)

    # ── 2. Препроцессинг ─────────────────────────────────────────────────────
    df, vocab_sizes = encode_categoricals(df)
    df, amt_stats   = normalize_amount(df)
    df["addr1"]     = df["addr1"].fillna(0.0)

    print(f"\nVocab sizes: {vocab_sizes}")

    if check_only:
        print("\n[--check] Предпросмотр данных (окна не строятся):")
        print(df.head())
        print(f"Колонки: {df.columns.tolist()}")
        return

    # ── 3. Построение windows ─────────────────────────────────────────────────
    print(f"\nСтроим sliding windows (seq_len={config.seq_len})...")
    data = build_windows(df, config.seq_len)
    N = len(data["label"])
    print(f"  Всего окон: {N:,}  |  фрод: {data['label'].sum():,} ({data['label'].mean()*100:.2f}%)")

    # ── 4. Time-based split ──────────────────────────────────────────────────
    print("\nРазбиваем по времени (train/val/test)...")
    train, val, test = time_based_split(data, config.train_frac, config.val_frac)
    save_split(train, config.processed_dir, "train")
    save_split(val,   config.processed_dir, "val")
    save_split(test,  config.processed_dir, "test")

    # ── 5. Метаданные ────────────────────────────────────────────────────────
    metadata = {
        "vocab_sizes": vocab_sizes,
        "amt_stats":   amt_stats,
        "seq_len":     config.seq_len,
        "static_dim":  STATIC_DIM,
        "n_train":     int(len(train["label"])),
        "n_val":       int(len(val["label"])),
        "n_test":      int(len(test["label"])),
    }
    meta_path = config.processed_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Готово! Данные сохранены в '{config.processed_dir}'")
    print(f"  metadata.json: {metadata}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Только проверка структуры данных, без построения окон")
    args = parser.parse_args()
    main(check_only=args.check)

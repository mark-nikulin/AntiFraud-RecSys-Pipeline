#!/usr/bin/env bash
# download_data.sh — скачивает IEEE-CIS Fraud Detection через Kaggle CLI
#
# Предусловие:
#   1. pip install kaggle
#   2. Положи ~/.kaggle/kaggle.json (API token с kaggle.com → Account → API)
#   3. chmod 600 ~/.kaggle/kaggle.json
#
# Запуск: bash scripts/download_data.sh

set -e

COMPETITION="ieee-fraud-detection"
DATA_DIR="data/raw"

echo "==> Создаём директорию $DATA_DIR..."
mkdir -p "$DATA_DIR"

echo "==> Скачиваем датасет '$COMPETITION' через kaggle CLI..."
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

echo "==> Распаковываем архив..."
unzip -q "$DATA_DIR/${COMPETITION}.zip" -d "$DATA_DIR"
rm "$DATA_DIR/${COMPETITION}.zip"

echo ""
echo "✓ Готово! Файлы в $DATA_DIR:"
ls -lh "$DATA_DIR"

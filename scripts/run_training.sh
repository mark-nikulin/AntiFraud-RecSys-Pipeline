#!/usr/bin/env bash
# run_training.sh — удобный запуск полного пайплайна из корня проекта
#
# Использование:
#   bash scripts/run_training.sh

set -e
cd "$(dirname "$0")/.."

echo "=== [1/2] Препроцессинг данных ==="
python src/prepare_data.py

echo ""
echo "=== [2/2] Обучение модели ==="
python src/train.py

echo ""
echo "✓ Всё готово!"
echo "  Мониторинг: tensorboard --logdir runs/"

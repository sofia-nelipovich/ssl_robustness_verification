# ssl_robustness_verification

Этот репозиторий содержит код и эксперименты по проверке устойчивости self-supervised learning (SSL) к дисбалансу классов на реальных (натуральных) long-tail датасетах. Работа вдохновлена статьёй *“Self-supervised Learning is More Robust to Dataset Imbalance”* (Liu et al., 2022) и переносит их идеи с CIFAR/ImageNet на специализированные домены (растения, животные).

---

## Цели проекта

- Проверить, сохраняется ли преимущество SSL над supervised learning (SL) в условиях дисбаланса классов на реальных датасетах (AgriNet, AwA), а не на синтетически-имбалансных версиях CIFAR/ImageNet. 
- Сравнить:
  - SL baseline, обученный на имбалансном датасете.
  - SSL с предобучением на ImageNet и последующим fine-tuning.
  - SSL, предобученный непосредственно на целевом датасете и оцененный через linear probe / fine-tuning.

---

## Структура репозитория

```
ssl_robustness_verification/
    README.md 
    project_tree.py        # Скрипт для вывода дерева проекта

visualization/
    analysis.ipynb         # Ноутбук с анализом результатов:
                           # - агрегация метрик
                           # - head/medium/tail accuracy
                           # - t-SNE визуализации

SL_Baselines/
    dataset.py             # Загрузка и подготовка датасетов для SL baseline
    main.py                # Точка входа для запуска supervised обучения
    trainer.py             # Цикл обучения, валидация, сохранение чекпоинтов
    utils.py               # Вспомогательные функции (метрики)

datasets/
    dataset.py             # Общие датасет-классы, используемые в разных скриптах

models/
    backbone_training.ipynb # Эксперименты по обучению/отладке backbone'ов в ноутбуке

Finetune/
    configs.py             # Конфиги для fine-tuning (гиперпараметры, пути к ckpt)
    dataset.py             # Датасеты для этапа fine-tuning (train/val/test split)
    main.py                # Точка входа для fine-tuning SSL/SL backbone'ов
    model.py               # Определение моделей (backbone + классификатор)
    trainer.py             # Логика обучения/валидации на этапе fine-tune
    utils.py               # Вспомогательные функции (логирование, метрики, чекпоинты)

Backbone/
    configs.py             # Конфиги для SSL pretraining backbone'ов
    dataset.py             # Датасеты/аугментации для pretraining (MoCo/SimSiam)
    model.py               # Определение архитектуры backbone'а (ResNet + SSL-головы)
    train.py               # Точка входа для SSL pretraining (MoCo/SimSiam)
    utils.py               # Вспомогательные функции для обучения backbone'а
```

---

## Запуск экспериментов

### 1. SSL pretraining (Backbone)

Папка `Backbone/` отвечает за self-supervised pretrain (MoCo/SimSiam) на ImageNet или на целевом датасете (AgriNet/AwA).

### 2. Supervised baseline (SL_Baselines)

Папка `SL_Baselines/` содержит код для supervised обучения на тех же датасетах.

`dataset.py` отвечает за загрузку разбиений train/val/test, `trainer.py` — за цикл обучения и валидации.

### 3. Fine-tuning SSL/SL backbone'ов (Finetune)

Папка `Finetune/` используется для дообучения предобученных backbone'ов (как SSL, так и SL) на целевом дисбалансном датасете.

В `model.py` собирается модель вида:
- `backbone` (замороженный или обучаемый),
- `classifier` (линейная или небольшая голова).

В `trainer.py` реализован:
- режим linear probe (обучаем только классификатор),
- режим full fine-tune (обучаем backbone + классификатор).

---

## Анализ результатов

Основной анализ делается в ноутбуке `visualization/analysis.ipynb`, где:
- считаются:
  - overall accuracy,
  - per-class accuracy,
  - head/medium/tail accuracy (по числу обучающих примеров на класс);
- строятся:
  - bar chart'ы (overall / head / medium / tail для разных методов),
  - t-SNE визуализации для feature-пространства разных моделей, чтобы наглядно увидеть, как разделяются классы.

---

## Идея экспериментов

- Сравниваются три сценария:
  1. Supervised обучение на дисбалансном датасете (SL baseline).
  2. SSL backbone, предобученный на ImageNet, + fine-tuning на дисбалансном целевом датасете.
  3. SSL backbone, предобученный напрямую на целевом датасете (in-domain), + linear probe / fine-tune.
- Цель — понять, сохраняется ли меньший “gap” между balanced/imbalanced обучением для SSL, как это показано Liu et al. на CIFAR/ImageNet, когда переходят к реальным long-tail доменам (растения, животные).

--- 
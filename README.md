# Предсказание сердечно-сосудистых заболеваний

![](docs/images/image1.jpg)

Бинарная классификация наличия болезни сердца на
[датасете UCI Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease)
(303 пациента, 13 признаков). Семь классификаторов обучаются на одном и том же
разбиении, лучший по выбранной метрике сохраняется как готовый sklearn Pipeline.

> Production-рефакторинг проекта [ShubhankarRawat/Heart-Disease-Prediction](https://github.com/ShubhankarRawat/Heart-Disease-Prediction).
> Исходный анализ и выбор датасета: Shubhankar Rawat.

## Быстрый старт

Требования: Python 3.12 (см. `.python-version`), [Poetry](https://python-poetry.org/) 2.x,
на macOS также `brew install libomp` (нужен для LightGBM/XGBoost).

```bash
make install   # создать .venv, установить зависимости и pre-commit хуки
make train     # обучить все модели -> models/model.joblib, reports/metrics.json
make eda       # сохранить EDA-графики -> reports/figures/
make check     # линтер + проверка типов + тесты (то же, что в CI)
```

Обучить часть моделей и выбрать лучшую по recall:

```bash
poetry run python -m heart_disease.train --models logistic_regression naive_bayes --metric recall
```

Использовать сохранённую модель:

```python
import joblib
from heart_disease.data import clean_data, load_raw_data, split_features_target

pipeline = joblib.load("models/model.joblib")
features, _ = split_features_target(clean_data(load_raw_data()))
pipeline.predict(features)  # сырые признаки: импутация и масштабирование внутри пайплайна
```

## Автоматическое переобучение и версии модели

В проекте есть локальный процесс с проверками допуска: новые размеченные
CSV поступают в `data/incoming`, алгоритм выбирается по групповой
кросс-валидации, а качество проверяется на отдельной новой выборке.
Кандидат заменяет действующую версию только после прохождения порогов
из [retraining.toml](retraining.toml), сравнения с ней и smoke-проверок.
Полная схема, формат CSV и условия отказа — в
[Model_retrain_BPMN.md](Model_retrain_BPMN.md).

```bash
make retrain                              # один защищённый запуск
poetry run python -m heart_disease.launchd          # создать plist локального расписания
poetry run python -m heart_disease.launchd --install # активировать расписание macOS
poetry run python -m heart_disease.predict --input features.csv --output predictions.csv
make rollback                             # восстановить предыдущую версию
```

После установки LaunchAgent процесс запускается 1-го числа месяца в 03:00
по часовому поясу компьютера, при поступлении CSV и при загрузке агента.
При недостатке данных или провале качества активная версия не меняется.
Начальное обучение может завершиться отказом: наличие модели в эксплуатации
не гарантируется до прохождения всех проверок.

Версии и отчёты сохраняются локально в `models/registry` и `models/runs`.
Для применения автоматически активируемой модели используйте
`heart-disease-predict` или `ActiveModel` из `heart_disease.registry`:

```python
from pathlib import Path
from heart_disease.registry import ActiveModel

model = ActiveModel(Path("models"))
predictions = model.predict(features)  # те же 13 признаков, без target
```

Этот потребитель автоматически подхватывает переключение `models/active.json`.
Пример `joblib.load("models/model.joblib")` выше относится к прежнему
экспериментальному `make train`, который сохранён отдельно.

## Структура проекта

```
├── data/raw/cleveland.csv      # исходный датасет (CSV без заголовка)
├── src/heart_disease/
│   ├── config.py               # пути, схема данных, константы
│   ├── data.py                 # загрузка и очистка данных
│   ├── models.py               # реестр моделей, sklearn Pipeline
│   ├── evaluate.py             # метрики
│   ├── train.py                # CLI: heart-disease-train
│   └── plots.py                # CLI: heart-disease-eda
├── tests/                      # тесты pytest (синтетические данные, запись только во tmp)
├── .pre-commit-config.yaml     # ruff, mypy, poetry check, гигиена файлов
├── .github/workflows/ci.yml    # CI: pre-commit + тесты
├── pyproject.toml / poetry.lock
└── Makefile
```

## Команды разработки

| Команда | Что делает |
|---|---|
| `make install` | Создаёт `.venv`, ставит зависимости из `poetry.lock`, устанавливает git-хуки |
| `make lint` | Линтер ruff |
| `make format` | Автоисправление и форматирование кода |
| `make typecheck` | Проверка типов mypy (strict) |
| `make test` | Тесты с отчётом о покрытии (минимум 80%) |
| `make check` | Все проверки разом, как в CI |
| `make clean` | Удаляет кэши и графики; сохраняет модели и историю оценочных выборок |

Виртуальное окружение `.venv` создаётся внутри проекта и в git не попадает.
Воспроизводимость обеспечивают три файла: `pyproject.toml` (зависимости),
`poetry.lock` (точные версии) и `.python-version` (версия Python).

## Результаты

Тестовая выборка 20%, стратифицированная, `random_state=0`:

| Модель | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Naive Bayes | 0.918 | 0.929 | 0.912 | 0.932 |
| SVM (RBF) | 0.852 | 0.821 | 0.836 | — |
| Logistic Regression | 0.836 | 0.821 | 0.821 | 0.944 |
| LightGBM | 0.836 | 0.750 | 0.808 | 0.918 |
| Random Forest | 0.820 | 0.750 | 0.793 | 0.880 |
| XGBoost | 0.820 | 0.714 | 0.784 | 0.909 |
| Decision Tree | 0.754 | 0.679 | 0.717 | 0.748 |

В тестовой выборке 61 пациент: одно предсказание — это ~1.6% accuracy,
поэтому разница между моделями в несколько пунктов находится в пределах шума.
Для медицинской задачи особенно важен recall — доля больных, которых модель не пропустила.

## Датасет

| Колонка | Описание |
|---|---|
| age | Возраст, лет |
| sex | Пол: 1 — мужской, 0 — женский |
| cp | Тип боли в груди: 1 типичная стенокардия, 2 атипичная, 3 неангинозная, 4 бессимптомная |
| trestbps | Артериальное давление в покое, мм рт. ст. |
| chol | Холестерин в сыворотке, мг/дл |
| fbs | Сахар натощак > 120 мг/дл (1 — да) |
| restecg | ЭКГ в покое: 0 норма, 1 аномалия ST-T, 2 гипертрофия левого желудочка |
| thalach | Максимальный пульс при нагрузке |
| exang | Стенокардия при нагрузке (1 — да) |
| oldpeak | Депрессия сегмента ST при нагрузке относительно покоя |
| slope | Наклон сегмента ST на пике нагрузки: 1 восходящий, 2 плоский, 3 нисходящий |
| ca | Число крупных сосудов, окрашенных при флюороскопии (0–3), 4 пропуска |
| thal | Талассемия: 3 норма, 6 фиксированный дефект, 7 обратимый дефект, 2 пропуска |
| target | 0 — нет болезни, 1–4 — есть (приводится к 0/1) |

## Что изменено по сравнению с оригиналом

- Один скрипт на 230 строк → устанавливаемый пакет с CLI-командами
- **Исправлена утечка данных (data leakage)**: пропуски заполнялись средним по всему датасету до train/test split; теперь импутация внутри пайплайна и считается только на train
- **Исправлен LightGBM**: `lgb.train(params={})` обучал регрессор с ручным порогом 0.5; заменён на `LGBMClassifier`
- Перепутанный порядок аргументов `confusion_matrix(y_pred, y_test)` и ручной подсчёт accuracy → `sklearn.metrics`, добавлены precision / recall / F1 / ROC-AUC
- Фиксированный `random_state` у всех моделей, стратифицированное разбиение
- `plt.show()` посреди обучения → графики сохраняются в файлы отдельной командой
- Исправлено описание датасета: в оригинальном README указано 779 записей и 15 колонок, фактически 303 и 14
- Poetry с lock-файлом, ruff, mypy в strict-режиме, pre-commit, pytest (покрытие 96%), CI на GitHub Actions

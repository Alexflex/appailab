# Прикладной искусственный интеллект: практические занятия

Учебный комплект предназначен для магистрантов начального уровня и содержит
первые три эталонных практических занятия по дисциплине "Прикладной
искусственный интеллект".

## Состав первого этапа

1. `notebooks/student/` - студенческие версии Jupyter Notebook
   (интерактивный вычислительный блокнот).
2. `notebooks/teacher/` - версии Jupyter Notebook для преподавателя.
3. `data/processed/` - учебные CSV-файлы (comma-separated values, текстовые
   таблицы со значениями, разделенными запятыми) для занятий 1-3.
4. `src/appai_lab/` - генераторы данных.
5. `scripts/generate_datasets.py` - пересоздание CSV-файлов.
6. `scripts/build_notebooks.py` - пересборка блокнотов.
7. `docs/teacher_guides/` - методические указания.
8. `docs/templates/` - шаблоны отчетов.
9. `docs/sources/` - проверенные источники и обзоры наборов данных.

## Установка локального окружения

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-base.txt
python -m ipykernel install --user --name appai-lab --display-name "Python (appai-lab)"
```

Файл `requirements-extended.txt` предназначен для последующих занятий, где
потребуются `pandapower`, `seaborn` и локальная LLM через Ollama. Полный набор
зависимостей курса можно установить командой `pip install -r requirements.txt`.

Если окружение уже создано, достаточно активировать его и обновить зависимости:

```bash
source venv/bin/activate
pip install -r requirements-base.txt
```

## Генерация учебных данных

```bash
source venv/bin/activate
python scripts/generate_datasets.py
```

Будут созданы файлы:

1. `data/processed/practice_01_motor_measurements.csv`;
2. `data/processed/practice_02_motor_efficiency_features.csv`;
3. `data/processed/practice_02_motor_efficiency_diagnostics.csv`;
4. `data/processed/practice_03_drive_mode_features.csv`;
5. `data/processed/practice_03_drive_mode_diagnostics.csv`;
6. `data/processed/practice_02_motor_efficiency.csv` - полный
   диагностический файл для совместимости;
7. `data/processed/practice_03_drive_mode_classification.csv` - полный
   диагностический файл для совместимости;
8. `data/processed/practice_01_03_dataset_catalog.csv`;
9. `data/processed/practice_01_03_dataset_assignments.csv`.

Файлы с суффиксом `_features.csv` используются в базовых студенческих
моделях. Файлы с суффиксом `_diagnostics.csv` применяются для объяснения
физических расчетов, контроля ограничений и демонстрации утечки данных.

## Сборка блокнотов

```bash
source venv/bin/activate
python scripts/build_notebooks.py
```

Для быстрой проверки инвариантов курса выполните:

```bash
source venv/bin/activate
python scripts/check_course_invariants.py
```

## Запуск Jupyter Notebook

```bash
source venv/bin/activate
jupyter lab
```

В интерфейсе JupyterLab выберите ядро `Python (appai-lab)`.

## Запуск в Google Colab

Для Google Colab (облачная среда выполнения Jupyter-блокнотов) рекомендуется
загрузить весь каталог проекта или клонировать репозиторий, затем открыть
студенческий блокнот из `notebooks/student/`. Блокноты ожидают, что рядом с
ними доступны каталоги `data` и `src`.

## Методическое решение по данным

Учебные CSV-файлы являются синтетическими, но построены по физически
осмысленным зависимостям и сопоставлены с открытыми реальными наборами данных.
Реальные источники перечислены в `docs/sources/SOURCES.md` и подробно
рассмотрены в `docs/sources/datasets_01_03_research.md`.

Дополнительно для занятий 1-3 подготовлены:

1. `docs/sources/dataset_assignment_details.md` - подробное описание всех
   найденных наборов данных, структуры признаков, целей, рисков утечки данных
   и развернутых заданий;
2. `docs/teacher_guides/visualization_and_explanation_guide_01_03.md` -
   руководство по объяснению графиков, схем и типовых ошибок интерпретации.

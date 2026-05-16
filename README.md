# Прикладной искусственный интеллект: практические занятия

Учебный комплект предназначен для магистрантов начального уровня и содержит
первые девять эталонных практических занятий по дисциплине "Прикладной
искусственный интеллект".

## Состав реализованных этапов

1. `notebooks/student/` - студенческие версии Jupyter Notebook
   (интерактивный вычислительный блокнот).
2. `notebooks/teacher/` - версии Jupyter Notebook для преподавателя.
3. `data/processed/` - учебные CSV-файлы (comma-separated values, текстовые
   таблицы со значениями, разделенными запятыми) для занятий 1-9.
4. `src/appai_lab/` - генераторы данных.
5. `scripts/generate_datasets.py` - пересоздание CSV-файлов.
6. `scripts/build_notebooks.py` - пересборка блокнотов занятий 1-3.
7. `scripts/build_notebooks_04_06.py` - пересборка блокнотов занятий 4-6.
8. `scripts/build_notebooks_07_09.py` - пересборка блокнотов занятий 7-9.
9. `scripts/prepare_external_datasets.py` - загрузка открытых внешних
   наборов данных и подготовка компактных учебных фрагментов.
10. `scripts/build_external_dataset_notebooks.py` - сборка расширенных
   блокнотов по реальным и открытым данным.
11. `docs/teacher_guides/` - методические указания.
12. `docs/templates/` - шаблоны отчетов.
13. `docs/sources/` - проверенные источники и обзоры наборов данных.

## Установка локального окружения

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-base.txt
python -m ipykernel install --user --name appai-lab --display-name "Python (appai-lab)"
```

Для занятий 8-9 требуется `pandapower` - библиотека моделирования
электроэнергетических систем на языке Python. В Google Colab эта зависимость
устанавливается из `requirements-colab.txt`. Для локального запуска занятий
1-9 можно установить базовые и расширенные зависимости:

```bash
pip install -r requirements-base.txt
pip install -r requirements-extended.txt
```

Полный набор зависимостей курса можно установить командой
`pip install -r requirements.txt`.

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
8. `data/processed/practice_04_haps_thermal_features.csv`;
9. `data/processed/practice_04_haps_thermal_diagnostics.csv`;
10. `data/processed/practice_04_haps_thermal.csv`;
11. `data/processed/practice_05_partial_discharge_features.csv`;
12. `data/processed/practice_05_partial_discharge_diagnostics.csv`;
13. `data/processed/practice_05_partial_discharge.csv`;
14. `data/processed/practice_06_equipment_modes_features.csv`;
15. `data/processed/practice_06_equipment_modes_diagnostics.csv`;
16. `data/processed/practice_06_equipment_modes.csv`;
17. `data/processed/practice_07_pd_signal_features.csv`;
18. `data/processed/practice_07_pd_signal_diagnostics.csv`;
19. `data/processed/practice_07_pd_signal_waveforms.csv`;
20. `data/processed/practice_08_power_flow_features.csv`;
21. `data/processed/practice_08_power_flow_diagnostics.csv`;
22. `data/processed/practice_08_power_flow_scenarios.csv`;
23. `data/processed/practice_09_power_flow_comparison_features.csv`;
24. `data/processed/practice_09_power_flow_comparison_diagnostics.csv`;
25. `data/processed/practice_09_power_flow_comparison.csv`;
26. `data/processed/practice_01_03_dataset_catalog.csv`;
27. `data/processed/practice_01_03_dataset_assignments.csv`;
28. `data/processed/practice_04_06_dataset_catalog.csv`;
29. `data/processed/practice_04_06_dataset_assignments.csv`;
30. `data/processed/practice_07_09_dataset_catalog.csv`;
31. `data/processed/practice_07_09_dataset_assignments.csv`.

Файлы с суффиксом `_features.csv` используются в базовых студенческих
моделях. Файлы с суффиксом `_diagnostics.csv` применяются для объяснения
физических расчетов, контроля ограничений и демонстрации утечки данных.

## Загрузка открытых внешних данных

Для расширенных занятий 1-3 используются три открытых источника с прямой
загрузкой без Kaggle:

1. `ElectricMotorTemperature`, Zenodo TSML Archive - многомерные временные
   ряды температуры электродвигателя, DOI `10.5281/zenodo.11235562`;
2. `Zenodo PMSM inverter fault diagnosis` - данные диагностики инвертора
   постоянно-магнитной синхронной машины (Permanent Magnet Synchronous Motor,
   PMSM);
3. `Processed Data for EV Powertrain Efficiency`, Mendeley Data - данные
   эффективности электропривода транспортного средства.

Для загрузки исходных данных в `data/raw/` и подготовки компактных таблиц в
`data/processed/external/` выполните:

```bash
source venv/bin/activate
python scripts/prefetch_data.py
```

`scripts/prefetch_data.py` предназначен для преподавателя: он выполняет
загрузку с повторными попытками, проверяет SHA256-контрольные суммы исходных
архивов и затем формирует компактные учебные CSV. Непосредственно на занятии
студентам скачивать полные внешние архивы не требуется.

Скрипт создает для каждого источника три файла:

1. `*_features.csv` - строгие признаки и целевая переменная классификации
   `is_allowed`; целевые и прокси-целевые величины регрессии вынесены в
   diagnostics-CSV;
2. `*_diagnostics.csv` - диагностические и производные величины для
   методического разбора;
3. `*_metadata.md` - краткое описание источника, лицензии, структуры данных
   и методических ограничений.

## Сборка блокнотов

```bash
source venv/bin/activate
python scripts/build_notebooks.py
python scripts/build_notebooks_04_06.py
python scripts/build_notebooks_07_09.py
```

Первая команда собирает базовые блокноты занятий 1-3, вторая - базовые
блокноты занятий 4-6, третья - базовые блокноты занятий 7-9.

Расширенные блокноты по внешним данным собираются отдельной командой:

```bash
source venv/bin/activate
python scripts/build_external_dataset_notebooks.py
```

После выполнения создаются 18 блокнотов:

1. `notebooks/external/student/` - 9 студенческих блокнотов без готовых
   выводов;
2. `notebooks/external/teacher/` - 9 преподавательских блокнотов, которые
   можно выполнить и сохранить с эталонными таблицами и графиками.

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

Google Colab - облачная среда выполнения Jupyter-блокнотов. Студенческие
блокноты содержат первую ячейку `COLAB_BOOTSTRAP_APPailab`, которая
автоматически определяет Colab, клонирует репозиторий, устанавливает
минимальные зависимости из `requirements-colab.txt` и переводит рабочий
каталог в корень проекта.

Для студентов подготовлены отдельные инструкции:

1. `docs/student_colab_guide.md` - подробная пошаговая инструкция;
2. `docs/student_colab_guide.pdf` - PDF-версия подробной инструкции;
3. `docs/student_colab_quickstart.md` - краткая памятка;
4. `docs/student_colab_quickstart.pdf` - одностраничная PDF-памятка.

PDF-файлы инструкций собираются из Markdown командой
`venv/bin/python scripts/build_student_colab_guides.py`. Для качественной
верстки используется `pandoc` и движок `XeLaTeX`; это позволяет корректно
обрабатывать кириллицу, таблицы, гиперссылки и переносы строк.

Важное условие доступа: прямые Colab-ссылки работают только тогда, когда
репозиторий `https://github.com/Alexflex/appailab` доступен студенту через
GitHub. Для массового запуска в аудитории рекомендуется использовать публичный
репозиторий или отдельный публичный учебный репозиторий без преподавательских
решений и закрытых данных. Если репозиторий остается приватным, каждый студент
должен иметь GitHub-доступ к нему и открыть блокнот в Colab через подключенную
учетную запись GitHub.

Базовые блокноты можно открыть напрямую:

| Занятие | Colab-ссылка |
|---|---|
| 01. Первичный анализ инженерных данных | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/01_engineering_data_student.ipynb) |
| 02. Регрессия КПД | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/02_motor_regression_student.ipynb) |
| 03. Дерево решений | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/03_drive_decision_tree_student.ipynb) |
| 04. Прогноз температуры электропривода HAPS | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/04_haps_thermal_modeling_student.ipynb) |
| 05. Классификация частичных разрядов | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/05_partial_discharge_classification_student.ipynb) |
| 06. Кластеризация режимов оборудования | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/06_equipment_modes_clustering_student.ipynb) |
| 07. Анализ сигналов частичных разрядов | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/07_pd_signal_analysis_student.ipynb) |
| 08. Расчет режима в pandapower | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/08_pandapower_power_flow_student.ipynb) |
| 09. Сравнение методов расчета режима | [Открыть](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/student/09_power_flow_comparison_student.ipynb) |

Расширенные блокноты по внешним данным:

| Источник | Занятие 01 | Занятие 02 | Занятие 03 |
|---|---|---|---|
| Mendeley EV Powertrain | [01](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/01_mendeley_ev_student.ipynb) | [02](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/02_mendeley_ev_student.ipynb) | [03](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/03_mendeley_ev_student.ipynb) |
| Zenodo PMSM Inverter | [01](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/01_zenodo_inverter_student.ipynb) | [02](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/02_zenodo_inverter_student.ipynb) | [03](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/03_zenodo_inverter_student.ipynb) |
| Zenodo Motor Temperature | [01](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/01_zenodo_motor_temp_student.ipynb) | [02](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/02_zenodo_motor_temp_student.ipynb) | [03](https://colab.research.google.com/github/Alexflex/appailab/blob/main/notebooks/external/student/03_zenodo_motor_temp_student.ipynb) |

Порядок работы в Colab:

1. Открыть нужный student-блокнот по ссылке.
2. Выполнить первую ячейку инициализации.
3. Выполнять остальные ячейки сверху вниз.
4. При необходимости сохранить личную копию блокнота в Google Drive.

Если прямая Colab-ссылка не открывается, используйте один из вариантов:

1. Сделать репозиторий публичным и повторно открыть ссылку.
2. Пригласить студентов в приватный репозиторий, затем открыть Colab,
   выбрать открытие блокнота из GitHub и авторизовать доступ к GitHub.
3. Запустить материалы на кафедральном JupyterHub/JupyterLab, если требуется
   закрытый режим без публикации GitHub-репозитория.

Полные raw-архивы внешних источников в Colab загружать не требуется:
компактные обработанные CSV уже находятся в `data/processed/` и
`data/processed/external/`.

## Методическое решение по данным

Учебные CSV-файлы являются синтетическими, но построены по физически
осмысленным зависимостям и сопоставлены с открытыми реальными наборами данных.
Реальные источники перечислены в `docs/sources/SOURCES.md` и подробно
рассмотрены в `docs/sources/datasets_01_03_research.md`,
`docs/sources/datasets_04_06_research.md` и
`docs/sources/datasets_07_09_research.md`.

Расширенные блокноты используют открытые внешние данные как отдельные
развернутые задания. При работе с временными рядами применяется групповое или
временное разбиение, а не случайное перемешивание соседних строк, поскольку
иначе возникает утечка информации между близкими во времени наблюдениями.

Дополнительно для занятий 1-3 подготовлены:

1. `docs/sources/dataset_assignment_details.md` - подробное описание всех
   найденных наборов данных, структуры признаков, целей, рисков утечки данных
   и развернутых заданий;
2. `docs/teacher_guides/visualization_and_explanation_guide_01_03.md` -
   руководство по объяснению графиков, схем и типовых ошибок интерпретации.
3. `docs/teacher_guides/visualization_and_explanation_guide_04_06.md` -
   руководство по визуализациям занятий 4-6.
4. `docs/teacher_guides/visualization_and_explanation_guide_07_09.md` -
   руководство по визуализациям занятий 7-9.

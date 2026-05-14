"""Сборка Jupyter Notebook для практических занятий 1-3.

Скрипт создает две версии каждого блокнота:
1. студенческую - с готовым кодовым шаблоном и заданиями для интерпретации;
2. преподавательскую - с тем же кодом, эталонными ориентирами и комментариями.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_STUDENT = PROJECT_ROOT / "notebooks" / "student"
NOTEBOOKS_TEACHER = PROJECT_ROOT / "notebooks" / "teacher"
COLAB_REPO_URL = "https://github.com/Alexflex/appailab.git"

REGRESSION_STRICT_FEATURES = [
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "temperature_c",
    "ambient_temp_c",
]
REGRESSION_LEAKAGE_DEMO_FEATURES = [
    *REGRESSION_STRICT_FEATURES,
    "current_a",
    "output_power_w",
    "loss_power_w",
]
CLASSIFICATION_STRICT_FEATURES = [
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "current_a",
    "ambient_temp_c",
    "temperature_c",
]
CLASSIFICATION_LEAKAGE_DEMO_FEATURES = [
    *CLASSIFICATION_STRICT_FEATURES,
    "efficiency",
    "output_power_w",
]


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def colab_bootstrap_cells() -> list[nbf.NotebookNode]:
    """Вернуть вводные ячейки для запуска student-блокнота в Google Colab."""

    return [
        md("""
## Инициализация среды Google Colab

Эта ячейка нужна только при запуске блокнота в Google Colab (облачная среда
выполнения Jupyter-блокнотов). Она клонирует репозиторий курса, устанавливает
минимальные зависимости и переводит рабочий каталог в корень проекта. При
локальном запуске или запуске на сервере кафедры ячейка не изменяет окружение.
"""),
        code(f"""
# COLAB_BOOTSTRAP_APPailab
from pathlib import Path
import os
import subprocess
import sys


REPO_URL = "{COLAB_REPO_URL}"
PROJECT_DIR = Path("/content/appailab")
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    if not PROJECT_DIR.exists():
        subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)
    os.chdir(PROJECT_DIR)

    src_dir = PROJECT_DIR / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    sentinel = PROJECT_DIR / ".colab_runtime_ready"
    requirements_file = PROJECT_DIR / "requirements-colab.txt"
    if not sentinel.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
            check=True,
        )
        sentinel.write_text("ok\\n", encoding="utf-8")

    required_csv = PROJECT_DIR / "data" / "processed" / "practice_01_motor_measurements.csv"
    if not required_csv.exists():
        subprocess.run([sys.executable, "scripts/generate_datasets.py"], check=True)

    print("Среда Google Colab подготовлена.")
    print("Корень проекта:", PROJECT_DIR)
else:
    print("Локальный или серверный запуск: инициализация Google Colab не требуется.")
"""),
    ]


def common_setup_code(dataset_filename: str, diagnostics_filename: str | None = None) -> str:
    diagnostics_file_line = (
        f'DIAGNOSTICS_FILE = DATA_DIR / "{diagnostics_filename}"'
        if diagnostics_filename is not None
        else "DIAGNOSTICS_FILE = None"
    )
    diagnostics_check = (
        " or (DIAGNOSTICS_FILE is not None and not DIAGNOSTICS_FILE.exists())"
        if diagnostics_filename is not None
        else ""
    )
    return f"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Поиск корня учебного проекта.
# Если блокнот запускается из корня репозитория, Path.cwd() уже указывает
# на нужный каталог. Если блокнот открыт из папки notebooks/student или
# notebooks/teacher, проверяются родительские каталоги.
candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
    Path.cwd().parent.parent.parent,
]

PROJECT_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src").exists() and (candidate / "data").exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise RuntimeError(
        "Не найден корень проекта. Запустите блокнот из каталога appai_lab "
        "или укажите путь к проекту вручную."
    )

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FILE = DATA_DIR / "{dataset_filename}"
{diagnostics_file_line}
CATALOG_FILE = DATA_DIR / "practice_01_03_dataset_catalog.csv"
ASSIGNMENTS_FILE = DATA_DIR / "practice_01_03_dataset_assignments.csv"

if not DATA_FILE.exists() or not CATALOG_FILE.exists() or not ASSIGNMENTS_FILE.exists(){diagnostics_check}:
    raise FileNotFoundError(
        "Не найдены учебные CSV-файлы. Выполните из корня проекта команду "
        "`python scripts/generate_datasets.py`, затем повторно запустите блокнот."
    )

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11

RANDOM_STATE = 20260507


def draw_process_diagram(labels, title):
    \"\"\"Построить простую схему последовательности этапов работы.

    Схема используется как учебная визуализация. Она не является алгоритмом,
    а показывает логическую структуру анализа: от источника данных к выводу.
    \"\"\"
    fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(labels)), 2.3))
    ax.set_axis_off()
    x_positions = np.linspace(0.06, 0.94, len(labels))

    for index, (x_pos, label) in enumerate(zip(x_positions, labels)):
        box = FancyBboxPatch(
            (x_pos - 0.075, 0.40),
            0.15,
            0.28,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="black",
            facecolor="#e8f1f8",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x_pos,
            0.54,
            label,
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
            transform=ax.transAxes,
        )
        if index < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x_pos + 0.085, 0.54),
                (x_positions[index + 1] - 0.085, 0.54),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.1,
                color="black",
                transform=ax.transAxes,
            )
            ax.add_patch(arrow)

    ax.set_title(title, fontsize=13, pad=14)
    plt.show()


def plot_correlation_heatmap(data, columns, title):
    \"\"\"Построить тепловую карту корреляций без дополнительных библиотек.\"\"\"
    corr = data[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Коэффициент корреляции")

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{{corr.iloc[i, j]:.2f}}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

DATA_FILE
"""


def dataset_assignment_cells(lesson_number: int) -> list[nbf.NotebookNode]:
    return [
        md(f"""
## Реестр найденных наборов данных и развернутые задания

В этом блокноте используются все найденные источники данных через единый
реестр. Реестр не загружает крупные внешние архивы автоматически. Он задает
отдельные расширенные задания, которые можно выполнять после базовой части
занятия.

Каждое задание включает:

1. теоретический блок - какие понятия и ограничения нужно объяснить;
2. практический блок - какие действия выполнить в Jupyter Notebook;
3. ожидаемые артефакты - какие таблицы, графики и выводы должны быть в отчете;
4. риски - какие ограничения источника необходимо учесть.

Ниже показаны все найденные наборы данных, а затем задания, относящиеся к
занятию {lesson_number}.
"""),
        code(f"""
dataset_catalog = pd.read_csv(CATALOG_FILE)
dataset_assignments = pd.read_csv(ASSIGNMENTS_FILE)

pd.set_option("display.max_colwidth", 120)

all_datasets_view = dataset_catalog[
    [
        "dataset_id",
        "name",
        "object",
        "license",
        "access",
        "size_note",
        "lessons",
        "risk_level",
        "implementation_status",
    ]
]
all_datasets_view
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

risk_counts = dataset_catalog["risk_level"].value_counts()
axes[0].bar(risk_counts.index, risk_counts.values, color="#4c78a8")
axes[0].set_title("Распределение наборов данных по уровню методического риска")
axes[0].set_ylabel("Число наборов")

lesson_counts = (
    dataset_assignments["lesson"]
    .astype(str)
    .value_counts()
    .sort_index()
)
axes[1].bar(lesson_counts.index, lesson_counts.values, color="#f58518")
axes[1].set_title("Число развернутых заданий по занятиям")
axes[1].set_xlabel("Номер занятия")
axes[1].set_ylabel("Число заданий")

plt.tight_layout()
plt.show()
"""),
        code(f"""
lesson_assignments = dataset_assignments[
    dataset_assignments["lesson"].astype(str) == "{lesson_number}"
].copy()

lesson_assignments[
    [
        "assignment_id",
        "assignment_title",
        "implementation_status",
        "dataset_structure",
        "minimum_working_subset",
        "target_rule",
        "split_rule",
        "theory_block",
        "practice_block",
        "recommended_visualizations",
        "expected_artifacts",
        "control_questions",
        "success_criteria",
        "risk_note",
    ]
]
"""),
        md("""
## Индивидуальное расширенное задание

Выберите один набор данных из таблицы выше и выполните соответствующее
развернутое задание. Если источник требует учетной записи, крупной загрузки
или специального формата файлов, допускается использовать малое заранее
подготовленное подмножество. В отчете обязательно укажите, была ли работа
выполнена на полном наборе данных, подмножестве или только на методическом
описании источника.
"""),
    ]


def source_section() -> str:
    return """
## Источники и проверка актуальности

1. ElectricMotorTemperature, Zenodo TSML Archive - открытый набор многомерных временных рядов для регрессии температуры электродвигателя. Используется как реальный ориентир структуры временных признаков. URL: https://zenodo.org/records/11235562
2. IEC 60034-1:2026. Rotating electrical machines - Part 1: Rating and performance. Используется как нормативный ориентир терминологии вращающихся электрических машин. URL: https://webstore.iec.ch/en/publication/89961
3. IEC 60034-2-1:2024. Rotating electrical machines - Part 2-1: Standard methods for determining losses and efficiency from tests. Используется как нормативный ориентир по потерям и КПД. URL: https://webstore.iec.ch/en/publication/67756
4. Документация pandas по пропущенным данным. URL: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
5. Документация scikit-learn по моделям, разбиению данных и метрикам качества. URL: https://scikit-learn.org/stable/

Дата проверки актуальности ссылок: 2026-05-07.
"""


def datasets_for_lessons_01_02_section() -> str:
    return """
## Рекомендуемые реальные наборы данных для занятий 1-2

Базовый учебный CSV в этом блокноте сохраняется для гарантированного запуска
в аудитории. При расширении работы его целесообразно сопоставлять с реальными
открытыми наборами данных:

1. ElectricMotorTemperature, Zenodo TSML Archive. URL:
   https://zenodo.org/records/11235562. Набор содержит открытые
   многомерные временные ряды (multivariate time series, многоканальные
   временные последовательности) для регрессии температуры электродвигателя.
   Применение: первичный анализ данных, регрессия температуры, обсуждение
   временной автокорреляции и группового разбиения. Ограничение: исходный
   формат `.ts` не содержит физических имен каналов, поэтому признаки в
   учебной таблице обозначаются нейтрально: `channel_00_mean`,
   `channel_01_mean` и далее.
2. Zenodo PMSM inverter fault diagnosis. URL:
   https://zenodo.org/records/14482932. Компактный набор измерений
   PMSM-инвертора с режимами отказов, фазными токами, напряжением
   звена постоянного тока и температурными признаками. Применение:
   первичный анализ, классификация и упрощенная регрессия температур.
   Ограничение: исходная постановка ближе к диагностике, чем к расчету КПД.
3. Processed Data for EV Powertrain Efficiency, Mendeley Data. URL:
   https://data.mendeley.com/datasets/kbwr2z8r3y. Набор
   предназначен для анализа эффективности электропривода транспортного
   средства. Применение: дополнительный пример перехода от траекторных
   данных к расчетной карте КПД. Ограничение: КПД является расчетным, а не
   стендовым измерением двигателя.

Методическое решение для эталонного занятия: использовать малый учебный CSV
как обязательный набор, а реальные источники - как расширение для проектов,
докладов и самостоятельной доработки.
"""


def datasets_for_lesson_03_section() -> str:
    return """
## Рекомендуемые реальные наборы данных для занятия 3

Для дерева решений особенно полезны наборы, где класс режима или состояния
можно объяснить через физически интерпретируемые признаки: ток, температура,
вибрация, скорость, момент, нагрузка.

1. Paderborn University Bearing Data Center. URL:
   https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter.
   Реальные данные стенда
   электромеханического привода: синхронно измеренные токи двигателя,
   вибрации, скорость, момент, радиальная нагрузка и температура. Классы:
   исправные и поврежденные подшипники. Применение: классификация состояния
   и извлечение простых статистических признаков из сигналов. Ограничение:
   лицензия CC BY-NC 4.0, коммерческое использование запрещено без отдельного
   согласования.
2. Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating
   Machine, Mendeley Data. URL:
   https://data.mendeley.com/datasets/ztmf3m7h5x/6. Реальные данные вращающейся машины при нормальном
   состоянии, дефектах подшипников, несоосности и дисбалансе ротора.
   Применение: бинарная или многоклассовая классификация после расчета
   признаков по временным сигналам. Ограничение: исходные файлы требуют
   предварительного чтения форматов MAT и TDMS.
3. MOTOR FAULT DETECTION DATA, Figshare. URL:
   https://figshare.com/articles/dataset/MOTOR_FAULT_DETECTION_DATA/27216219.
   Набор по трехфазному асинхронному
   двигателю с синхронными сигналами вибрации, тока и напряжения, организован
   в CSV-файлы для исправных и неисправных режимов. Применение: удобная
   база для дерева решений после агрегации сигналов по окнам. Ограничение:
   общий объем около 5 ГБ, для аудиторного занятия нужна малая выборка.
4. Case Western Reserve University Bearing Data Center. URL:
   https://engineering.case.edu/bearingdatacenter/welcome. Классический
   эталонный сравнительный набор данных (benchmark dataset) по диагностике дефектов
   подшипников электродвигателя. Применение: демонстрация классификации
   нормального и поврежденного состояния. Ограничение: данные представлены
   как вибрационные сигналы, поэтому для начинающих нужна готовая процедура
   извлечения признаков.

Методическое решение для эталонного занятия: использовать компактный учебный
CSV с заранее рассчитанными признаками, а реальные наборы применять на
следующем уровне после освоения дерева решений.
"""


def notebook_01(teacher: bool) -> list[nbf.NotebookNode]:
    title_suffix = " Версия преподавателя" if teacher else ""
    engineering_feature_cell = (
        """
feature_columns = [
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "current_a",
    "ambient_temp_c",
    "temperature_c",
]
diagnostic_columns = [
    "winding_resistance_ohm",
    "output_power_w",
    "loss_power_w",
    "torque_limit_nm",
]
target_column = "efficiency"
"""
        if teacher
        else """
# TODO: заполните базовый набор измеряемых признаков.
# Не включайте `output_power_w`, `loss_power_w` и `efficiency`.
feature_columns = [
    # "speed_rpm",
    # "torque_nm",
    # "voltage_v",
    # "current_a",
    # "ambient_temp_c",
    # "temperature_c",
]
diagnostic_columns = [
    "winding_resistance_ohm",
    "output_power_w",
    "loss_power_w",
    "torque_limit_nm",
]
target_column = "efficiency"

if not feature_columns:
    raise ValueError("Заполните `feature_columns` перед продолжением анализа.")
"""
    )
    outlier_columns_cell = (
        """
outlier_columns = ["current_a", "temperature_c", "efficiency"]
outlier_summary = {}

for column in outlier_columns:
    mask = iqr_outlier_mask(df[column].dropna())
    outlier_summary[column] = int(mask.sum())

pd.Series(outlier_summary, name="candidate_outliers")
"""
        if teacher
        else """
# TODO: выберите не менее трех столбцов для поиска кандидатов в выбросы.
# Рекомендуется включить ток, температуру и КПД.
outlier_columns = [
    # "current_a",
    # "temperature_c",
    # "efficiency",
]

if len(outlier_columns) < 3:
    raise ValueError("Укажите не менее трех столбцов в `outlier_columns`.")

outlier_summary = {}

for column in outlier_columns:
    mask = iqr_outlier_mask(df[column].dropna())
    outlier_summary[column] = int(mask.sum())

pd.Series(outlier_summary, name="candidate_outliers")
"""
    )
    cells: list[nbf.NotebookNode] = [
        md(f"""
# Практическое занятие 1. Инженерные данные и постановка задачи прикладного искусственного интеллекта{title_suffix}

## Назначение занятия

Цель занятия - освоить первичный анализ инженерной таблицы данных на примере
маломощного высокооборотного электромеханического преобразователя на постоянных
магнитах.

Искусственный интеллект (Artificial Intelligence, AI) в данном занятии
понимается как совокупность методов извлечения закономерностей из данных.
Машинное обучение (Machine Learning, ML) - раздел искусственного интеллекта,
в котором модель настраивает параметры по наблюдениям.

К концу занятия студент должен уметь:

1. загрузить таблицу данных;
2. различить наблюдение, признак и целевую переменную;
3. обнаружить пропуски и возможные выбросы;
4. построить первичные графики;
5. сформулировать инженерный вывод с учетом единиц измерения.
"""),
        md("""
## Основные определения

Наблюдение - одна строка таблицы, соответствующая одному режиму работы.
Признак (feature) - входная величина, которую можно использовать для анализа
или обучения модели. Целевая переменная (target variable) - величина, которую
требуется объяснить, предсказать или классифицировать.

Коэффициент полезного действия, КПД, обозначается `efficiency` и отражает
отношение полезной выходной мощности к входной электрической мощности. В
учебной таблице КПД представлен как безразмерная величина от 0 до 1.
"""),
        md("""
## Как устроены модели первичного анализа данных

В занятии 1 не строится прогнозная модель машинного обучения. Тем не менее
используются формальные модели описания данных. Модель данных - это
упрощенное математическое представление объекта анализа. Для табличного
набора данных таким представлением является матрица признаков:

$$X = [x_{ij}], \\quad i = 1, ..., n, \\quad j = 1, ..., p,$$

где `n` - число наблюдений, `p` - число признаков, `x_ij` - значение признака
`j` в наблюдении `i`. Такая запись позволяет отделить физический объект
измерения от вычислительной формы, в которой он анализируется.

Описательная статистика (descriptive statistics) строит компактное описание
распределения признака. Среднее значение оценивает центральный уровень:

$$\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i.$$

Выборочное стандартное отклонение показывает типичный масштаб разброса:

$$s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})^2}.$$

Медиана - значение, которое делит упорядоченную выборку на две равные части.
Квартили - значения, отделяющие четверти упорядоченной выборки. Первый
квартиль `Q1` соответствует 25 %, третий квартиль `Q3` соответствует 75 %.
Межквартильный размах (interquartile range, IQR) определяется как

$$IQR = Q3 - Q1.$$

В учебном анализе кандидатами в выбросы считаются наблюдения за пределами
интервала

$$[Q1 - 1.5 \\cdot IQR, \\; Q3 + 1.5 \\cdot IQR].$$

Это правило не доказывает ошибочность наблюдения. В инженерных данных выброс
может быть следствием неисправности, переходного процесса или редкого, но
физически допустимого режима. Поэтому любое статистическое правило должно
сопоставляться с единицами измерения и физическими ограничениями объекта.

Корреляционный анализ оценивает согласованное изменение двух числовых
признаков. Коэффициент корреляции Пирсона (Pearson correlation coefficient)
вычисляется по формуле

$$r_{xy} =
\\frac{\\sum_{i=1}^{n}(x_i-\\bar{x})(y_i-\\bar{y})}
{\\sqrt{\\sum_{i=1}^{n}(x_i-\\bar{x})^2}\\sqrt{\\sum_{i=1}^{n}(y_i-\\bar{y})^2}}.$$

Значение `r_xy` находится в диапазоне от -1 до 1. Значение около 1 указывает
на сильную прямую линейную связь, значение около -1 - на сильную обратную
линейную связь, значение около 0 - на отсутствие выраженной линейной связи.
Корреляция не является доказательством причинно-следственной зависимости.
"""),
        md("""
## Физическая основа учебного набора данных

Учебная таблица имитирует структуру стендовых измерений постоянно-магнитной
синхронной машины (Permanent Magnet Synchronous Motor, PMSM). Такая машина
имеет ротор с постоянными магнитами и статорную обмотку, создающую вращающееся
магнитное поле. В практических занятиях PMSM используется как инженерный
пример электромеханического преобразователя: входом является электрическая
мощность, а полезным выходом - механическая мощность на валу.

Базовые расчетные соотношения:

$\\omega = 2 \\pi n / 60$,

$P_{out} = M \\omega$,

$P_{in} = U I$,

$\\eta = P_{out} / P_{in}$,

$P_{loss} = P_{in} - P_{out}$.

Здесь `omega_rad_s` - угловая скорость в радианах в секунду, `speed_rpm` -
частота вращения в оборотах в минуту, `torque_nm` - момент на валу,
`voltage_v` - напряжение, `current_a` - ток. Реальные испытания требуют учета
методики измерения потерь, теплового состояния и неопределенности датчиков.
Поэтому данный CSV следует рассматривать как воспроизводимый учебный набор:
он согласован с инженерной логикой открытых стендовых данных, но не заменяет
сертификационные испытания конкретного двигателя. В расчетных проверках
допустимо небольшое расхождение из-за округления сохраненных CSV-значений.
"""),
        md(source_section()),
        md(datasets_for_lessons_01_02_section()),
        code(common_setup_code("practice_01_motor_measurements.csv")),
        md("""
## Логическая схема занятия

Первичный анализ данных не начинается с обучения модели. Сначала необходимо
понять происхождение данных, физический смысл столбцов и ограничения
измерений. В инженерной задаче это особенно важно, потому что численно
правильная обработка может привести к физически неверному выводу, если
игнорировать единицы измерения, режим работы или происхождение признака.
"""),
        md("""
## Последовательность подготовки данных на этапе первичного анализа

Подготовка данных должна выполняться как воспроизводимая процедура. Это
означает, что каждый шаг должен быть описан так, чтобы другой исследователь
мог повторить обработку и получить тот же результат.

Рекомендуемая последовательность действий:

1. Зафиксировать источник данных: название файла, происхождение, дату
   получения, лицензию и ограничения использования.
2. Определить объект наблюдения: одна строка таблицы, измерительный профиль,
   экспериментальный запуск или окно временного сигнала.
3. Составить паспорт столбцов: имя столбца, физический смысл, единица
   измерения, роль в анализе, допустимость использования как признака.
4. Разделить столбцы на группы: служебные поля, измеряемые физические
   величины, расчетные величины, целевые переменные и диагностические метки.
5. Проверить типы данных: числовые признаки должны быть представлены как
   числа, категориальные признаки - как ограниченный набор классов, даты и
   время - как упорядоченные значения.
6. Проверить пропуски и зафиксировать решение: оставить как индикатор
   отсутствия измерения, удалить строку, заменить медианой или применить
   иной метод восстановления.
7. Проверить физические ограничения: знак величин, диапазон КПД, согласование
   мощности, допустимость момента, тока и температуры.
8. Найти кандидаты в выбросы статистическими методами и отдельно решить,
   являются ли они ошибками измерения или редкими инженерными режимами.
9. Построить первичные визуализации: распределения, диаграммы размаха,
   диаграммы рассеяния, тепловую карту корреляций.
10. Сформулировать инженерный вывод: какие данные пригодны для дальнейшего
    моделирования, какие ограничения требуют учета, какие признаки могут
    привести к утечке данных.

На этом занятии результатом является не обученная модель, а проверенная и
описанная таблица данных. Без такого этапа последующее моделирование может
дать численно высокие метрики, но методически ошибочные выводы.
"""),
        code("""
draw_process_diagram(
    [
        "Источник данных",
        "Таблица наблюдений",
        "Паспорт признаков",
        "Качество данных",
        "Визуализация",
        "Инженерный вывод",
    ],
    "Последовательность первичного анализа инженерных данных",
)
"""),
        code("""
df = pd.read_csv(DATA_FILE)

print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")
df.head()
"""),
        md("""
## Паспорт используемого учебного набора данных

Паспорт набора данных фиксирует происхождение, объект наблюдения, единицу
наблюдения и ограничения применимости. Такой паспорт нужен до построения
модели, поскольку он предотвращает ошибочную трактовку служебных столбцов,
расчетных величин и физически измеряемых величин как равноправных признаков.
"""),
        code("""
base_dataset_passport = pd.DataFrame(
    {
        "parameter": [
            "Файл",
            "Объект",
            "Наблюдение",
            "Объем",
            "Основная цель занятия",
            "Физически измеряемые величины",
            "Расчетные величины",
            "Служебные поля",
            "Учебные дефекты качества",
            "Ограничение применимости",
            "Энергетический баланс",
        ],
        "value": [
            DATA_FILE.name,
            "маломощный электромеханический преобразователь на постоянных магнитах",
            "один режим работы двигателя",
            f"{df.shape[0]} строк и {df.shape[1]} столбцов",
            "первичный анализ, проверка качества данных и постановка задачи",
            "скорость, момент, напряжение, ток, температуры",
            "выходная мощность, потери, КПД, предел момента",
            "sample_id, profile_id",
            "контролируемые пропуски и выбросы",
            "данные учебные; выводы нельзя переносить на конкретный двигатель без проверки",
            "мощность, потери и КПД согласованы с измеренными напряжением и током с учетом округления",
        ],
    }
)

base_dataset_passport
"""),
        md("""
## Описание столбцов

На первом этапе необходимо явно определить, какие столбцы являются признаками,
а какой столбец является целевой переменной. В данном занятии целевой
переменной для интерпретации выберем КПД `efficiency`.

Столбцы `output_power_w`, `loss_power_w`, `torque_limit_nm` и
`winding_resistance_ohm` рассматриваются как диагностические или производные.
Они полезны для проверки физической согласованности, но не входят в базовый
набор признаков. `winding_resistance_ohm` является расчетной оценкой
сопротивления обмотки по температуре с небольшим шумом измерения; при
одновременном использовании с `temperature_c` возможна сильная линейная
зависимость признаков, то есть коллинеарность (collinearity).
"""),
        code(engineering_feature_cell + """
column_description = pd.DataFrame(
    {
        "column": df.columns,
        "role": [
            "служебный идентификатор",
            "номер профиля измерения",
            "признак",
            "признак",
            "признак",
            "признак",
            "признак",
            "производный диагностический столбец",
            "признак",
            "расчетная величина",
            "расчетная величина",
            "целевая переменная",
            "инженерное ограничение",
        ],
        "unit": [
            "-",
            "-",
            "rpm",
            "N*m",
            "V",
            "A",
            "deg_C",
            "Ohm",
            "deg_C",
            "W",
            "W",
            "1",
            "N*m",
        ],
        "physical_meaning": [
            "номер наблюдения",
            "условный номер измерительного профиля",
            "частота вращения вала",
            "электромагнитный момент",
            "напряжение питания",
            "ток двигателя",
            "температура окружающей среды",
            "сопротивление обмотки при текущей температуре",
            "температура обмотки",
            "механическая выходная мощность",
            "потери мощности",
            "коэффициент полезного действия",
            "учебный предел момента для данной скорости",
        ],
        "use_note": [
            "не использовать как физический признак",
            "использовать для группировки или проверки профилей",
            "основной режимный признак",
            "основной режимный признак",
            "электрический признак",
            "электрический признак",
            "внешнее условие",
            "производный диагностический признак",
            "тепловой признак",
            "расчетная величина; проверять риск утечки",
            "расчетная величина; проверять риск утечки",
            "цель в занятиях 1-2",
            "не является измерением; используется для учебных ограничений",
        ],
    }
)
column_description
"""),
        code("""
role_counts = column_description["role"].value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(role_counts.index, role_counts.values, color="#4c78a8")
ax.set_ylabel("Число столбцов")
ax.set_title("Структура столбцов по методической роли")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
"""),
        md("""
## Проверка пропусков

Пропуск - отсутствующее значение в таблице. В pandas пропуски обычно
отображаются как `NaN`. Перед обучением модели пропуски необходимо
обнаружить и обработать: удалить строки, заменить медианой или применить
более сложный метод восстановления.
"""),
        md("""
## Стратегия обработки пропусков

Стратегия обработки пропусков (missing value strategy) должна выбираться до
построения модели. Сначала необходимо определить возможную причину пропуска:
отсутствие измерения, отказ датчика, невозможность расчета производной
величины или ошибка переноса данных. Только после этого выбирается действие.

Основные варианты обработки:

1. исключить наблюдение, если пропусков мало и удаление не искажает выборку;
2. заменить значение статистической оценкой, например медианой;
3. восстановить значение по физической или расчетной модели;
4. оставить пропуск и использовать его как индикатор качества измерения.

Медианная замена (median imputation) допустима в учебной задаче как простая и
воспроизводимая процедура. Однако она изменяет распределение признака и может
ослабить связь между физической величиной и целевой переменной. Поэтому в
отчете необходимо указать, какие столбцы были обработаны и почему выбран
именно этот способ.
"""),
        code("""
missing_table = (
    df.isna()
    .sum()
    .rename("missing_count")
    .to_frame()
    .assign(missing_share=lambda x: x["missing_count"] / len(df))
)
missing_table[missing_table["missing_count"] > 0]
"""),
        code("""
fig, ax = plt.subplots(figsize=(10, 4))
missing_matrix = df.isna().astype(int).T
ax.imshow(missing_matrix, aspect="auto", cmap="Greys")
ax.set_yticks(range(len(df.columns)))
ax.set_yticklabels(df.columns)
ax.set_xlabel("Номер наблюдения")
ax.set_title("Карта пропусков: черные элементы соответствуют отсутствующим значениям")
plt.tight_layout()
plt.show()
"""),
        md("""
## Контроль физической реализуемости

Перед интерпретацией графиков необходимо проверить простые физические
ограничения. Такие проверки не доказывают корректность набора данных, но
быстро выявляют грубые ошибки: отрицательную скорость, КПД вне диапазона от
0 до 1, неуникальные идентификаторы или несогласованность расчетной мощности.
"""),
        code("""
physical_checks = pd.Series(
    {
        "sample_id_unique": df["sample_id"].is_unique,
        "speed_positive": bool((df["speed_rpm"].dropna() > 0).all()),
        "voltage_positive": bool((df["voltage_v"].dropna() > 0).all()),
        "current_positive": bool((df["current_a"].dropna() > 0).all()),
        "efficiency_between_0_and_1": bool(df["efficiency"].dropna().between(0, 1).all()),
        "torque_not_above_limit_after_dropna": bool(
            (df.dropna(subset=["torque_nm"])["torque_nm"] <= df.dropna(subset=["torque_nm"])["torque_limit_nm"]).all()
        ),
    },
    name="check_passed",
)

physical_checks.to_frame()
"""),
        code("""
physics_df = df.dropna(subset=["speed_rpm", "torque_nm", "output_power_w", "voltage_v", "current_a", "efficiency"]).copy()
physics_df["omega_rad_s"] = 2.0 * np.pi * physics_df["speed_rpm"] / 60.0
physics_df["recomputed_output_power_w"] = physics_df["torque_nm"] * physics_df["omega_rad_s"]
physics_df["input_power_w"] = physics_df["voltage_v"] * physics_df["current_a"]
physics_df["recomputed_efficiency"] = physics_df["recomputed_output_power_w"] / physics_df["input_power_w"]
physics_df["output_power_relative_error"] = (
    (physics_df["recomputed_output_power_w"] - physics_df["output_power_w"]).abs()
    / physics_df["output_power_w"].clip(lower=1e-9)
)

physics_summary = physics_df[
    [
        "recomputed_output_power_w",
        "input_power_w",
        "recomputed_efficiency",
        "output_power_relative_error",
    ]
].describe().T

physics_summary
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(
    physics_df["output_power_w"],
    physics_df["recomputed_output_power_w"],
    alpha=0.70,
)
power_min = min(physics_df["output_power_w"].min(), physics_df["recomputed_output_power_w"].min())
power_max = max(physics_df["output_power_w"].max(), physics_df["recomputed_output_power_w"].max())
axes[0].plot([power_min, power_max], [power_min, power_max], color="black")
axes[0].set_xlabel("Мощность из таблицы, W")
axes[0].set_ylabel("Мощность по формуле M*omega, W")
axes[0].set_title("Проверка расчетной механической мощности")

profile_efficiency = [
    physics_df.loc[physics_df["profile_id"] == profile, "efficiency"]
    for profile in sorted(physics_df["profile_id"].unique())
]
axes[1].boxplot(profile_efficiency, labels=sorted(physics_df["profile_id"].unique()))
axes[1].set_xlabel("profile_id")
axes[1].set_ylabel("КПД")
axes[1].set_title("Распределение КПД по профилям измерений")

plt.tight_layout()
plt.show()
"""),
        md("""
## Первичная статистика

Описательная статистика позволяет проверить порядок величин, диапазоны и
возможные аномалии. Следует сопоставлять числа с физическим смыслом: ток
измеряется в амперах, момент - в Н*м, скорость - в оборотах в минуту.
"""),
        code("""
df[feature_columns + [target_column]].describe().T
"""),
        code("""
hist_columns = ["speed_rpm", "torque_nm", "current_a", "temperature_c", "efficiency"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.ravel()

for ax, column in zip(axes, hist_columns):
    ax.hist(df[column].dropna(), bins=24, color="#4c78a8", edgecolor="white")
    ax.set_title(column)
    ax.set_ylabel("Число наблюдений")

axes[-1].set_axis_off()
plt.suptitle("Распределения основных физических величин", y=1.02)
plt.tight_layout()
plt.show()
"""),
        md("""
## Визуальный анализ зависимостей

График рассеяния показывает, как пары величин связаны между собой. Он не
доказывает причинно-следственную связь, но помогает сформулировать инженерную
гипотезу.
"""),
        code("""
plot_df = df.dropna(subset=["torque_nm", "efficiency", "current_a", "temperature_c"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(plot_df["torque_nm"], plot_df["efficiency"], alpha=0.75)
axes[0].set_xlabel("Момент, Н*м")
axes[0].set_ylabel("КПД, доли единицы")
axes[0].set_title("КПД в зависимости от момента")

axes[1].scatter(plot_df["current_a"], plot_df["temperature_c"], alpha=0.75, color="tab:red")
axes[1].set_xlabel("Ток, А")
axes[1].set_ylabel("Температура, deg_C")
axes[1].set_title("Температура в зависимости от тока")

plt.tight_layout()
plt.show()
"""),
        code("""
speed_torque_df = df.dropna(subset=["speed_rpm", "torque_nm", "efficiency"])

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(
    speed_torque_df["speed_rpm"],
    speed_torque_df["torque_nm"],
    c=speed_torque_df["efficiency"],
    cmap="viridis",
    alpha=0.80,
)
ax.set_xlabel("Скорость, rpm")
ax.set_ylabel("Момент, N*m")
ax.set_title("Область режимов: скорость, момент и КПД")
fig.colorbar(scatter, ax=ax, label="КПД")
plt.tight_layout()
plt.show()
"""),
        md("""
## Поиск возможных выбросов

Выброс - наблюдение, резко отличающееся от основной массы данных. В
инженерных задачах выброс не всегда является ошибкой: это может быть
переходный режим, перегрузка или редкое рабочее состояние.

Ниже используется межквартильный размах (Interquartile Range, IQR). Это
разность между третьим и первым квартилем. Наблюдения вне интервала
`Q1 - 1.5 * IQR` и `Q3 + 1.5 * IQR` рассматриваются как кандидаты в выбросы.
"""),
        code("""
def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)
""" + outlier_columns_cell),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, column in zip(axes, outlier_columns):
    ax.boxplot(df[column].dropna(), vert=True)
    ax.set_title(column)
    ax.set_ylabel("Значение")

plt.suptitle("Диаграммы размаха для поиска кандидатов в выбросы", y=1.04)
plt.tight_layout()
plt.show()
"""),
        md("""
## Простейшая очистка данных

Для учебной задачи применим медианную замену пропусков. Медиана менее
чувствительна к выбросам, чем среднее арифметическое. В реальной работе
выбор метода обработки пропусков должен быть обоснован причиной их появления.
"""),
        md("""
## Предупреждение о пересчете целевой переменной

Пропуски в исходной таблице внесены после расчета производных энергетических
величин. Поэтому в отдельных строках может сохраняться `output_power_w`, хотя
соответствующий первичный столбец `torque_nm` содержит пропуск. Это намеренный
учебный артефакт: студент должен обнаружить, что физический баланс нельзя
проверять механически без анализа происхождения столбцов.

После медианной замены первичных величин `torque_nm`, `current_a` и
`temperature_c` значения `output_power_w`, `loss_power_w` и `efficiency`
пересчитываются заново. Следовательно, часть значений `efficiency` в
`clean_df` получена не из исходных измерений, а из восстановленных медианой
значений. Такую таблицу допустимо использовать для демонстрации очистки
данных, но нельзя без пояснений переносить в регрессионную задачу занятия 2.
"""),
        md("""
## Протокол преобразования данных

Протокол преобразования данных (data transformation protocol) фиксирует,
какие действия выполнены с исходной таблицей. Исходные значения не следует
перезаписывать без сохранения процедуры обработки. Воспроизводимый анализ
должен создавать очищенную копию данных и явно описывать все изменения.

Минимальный протокол включает:

1. перечень обработанных столбцов;
2. способ обработки пропусков;
3. правило работы с выбросами;
4. обоснование сохранения или исключения наблюдений;
5. проверку качества таблицы после преобразования.

В данном занятии выбросы не удаляются автоматически. Они рассматриваются как
кандидаты на дополнительную инженерную проверку, поскольку экстремальное
значение может соответствовать как ошибке измерения, так и редкому
допустимому режиму работы.
"""),
        code("""
clean_df = df.copy()

# Пропуски заменяются только в первичных измеряемых столбцах.
# Для torque_nm и current_a используется физически ограниченная медианная
# замена: восстановленное значение не должно приводить к КПД больше единицы.
# Это не делает восстановленные строки "истинными измерениями", но сохраняет
# базовые энергетические ограничения учебной таблицы.
primary_imputation_columns = ["torque_nm", "current_a", "temperature_c"]

original_missing = df[primary_imputation_columns].isna()
target_max_efficiency_after_imputation = 0.98

clean_df["temperature_c"] = clean_df["temperature_c"].fillna(clean_df["temperature_c"].median())

omega_rad_s_initial = 2.0 * np.pi * clean_df["speed_rpm"] / 60.0
median_torque = clean_df["torque_nm"].median()
median_current = clean_df["current_a"].median()

torque_missing_mask = clean_df["torque_nm"].isna()
if torque_missing_mask.any():
    torque_physical_cap = (
        target_max_efficiency_after_imputation
        * clean_df.loc[torque_missing_mask, "voltage_v"]
        * clean_df.loc[torque_missing_mask, "current_a"].fillna(median_current)
        / omega_rad_s_initial.loc[torque_missing_mask]
    )
    clean_df.loc[torque_missing_mask, "torque_nm"] = np.minimum(
        median_torque,
        torque_physical_cap,
    )

current_missing_mask = clean_df["current_a"].isna()
if current_missing_mask.any():
    output_power_for_current = clean_df.loc[current_missing_mask, "torque_nm"] * omega_rad_s_initial.loc[current_missing_mask]
    current_physical_floor = (
        output_power_for_current
        / (
            target_max_efficiency_after_imputation
            * clean_df.loc[current_missing_mask, "voltage_v"]
        )
    )
    clean_df.loc[current_missing_mask, "current_a"] = np.maximum(
        median_current,
        current_physical_floor,
    )

omega_rad_s = 2.0 * np.pi * clean_df["speed_rpm"] / 60.0
clean_df["output_power_w"] = clean_df["torque_nm"] * omega_rad_s
input_power_w = clean_df["voltage_v"] * clean_df["current_a"]
clean_df["loss_power_w"] = input_power_w - clean_df["output_power_w"]
clean_df["efficiency"] = clean_df["output_power_w"] / input_power_w

print("Число пропусков после обработки и пересчета:", int(clean_df.isna().sum().sum()))
clean_df[feature_columns + [target_column]].corr(numeric_only=True)[target_column].sort_values(ascending=False)
"""),
        code("""
imputation_audit = pd.DataFrame(
    {
        "column": primary_imputation_columns,
        "missing_before": [int(original_missing[column].sum()) for column in primary_imputation_columns],
        "missing_after": [int(clean_df[column].isna().sum()) for column in primary_imputation_columns],
    }
)
imputation_audit["method"] = [
    "медиана с физическим ограничением КПД",
    "медиана с физическим ограничением КПД",
    "медиана",
]
imputation_audit
"""),
        md("""
Физически ограниченная медианная замена отличается от простой медианной
замены тем, что восстановленные `torque_nm` и `current_a` дополнительно
проверяются по неравенству `efficiency <= 0.98`. Без такого ограничения
отдельные строки могут получить КПД больше единицы, потому что медианный
момент и медианный ток взяты из разных режимов работы.
"""),
        code("""
clean_balance_df = clean_df.copy()
clean_balance_df["omega_rad_s"] = 2.0 * np.pi * clean_balance_df["speed_rpm"] / 60.0
clean_balance_df["balance_output_power_w"] = (
    clean_balance_df["torque_nm"] * clean_balance_df["omega_rad_s"]
)
clean_balance_df["balance_relative_error"] = (
    (clean_balance_df["balance_output_power_w"] - clean_balance_df["output_power_w"]).abs()
    / clean_balance_df["output_power_w"].clip(lower=1e-9)
)

pd.Series(
    {
        "max_relative_error_after_cleaning": clean_balance_df["balance_relative_error"].max(),
        "mean_relative_error_after_cleaning": clean_balance_df["balance_relative_error"].mean(),
    },
    name="energy_balance_after_cleaning",
).to_frame("value")
"""),
        code("""
plot_correlation_heatmap(
    clean_df,
    feature_columns + [target_column],
    "Корреляции признаков после медианной замены пропусков",
)
"""),
        md("""
## Переход от первичного анализа к постановке задачи машинного обучения

После первичного анализа необходимо зафиксировать, какая задача машинного
обучения может быть поставлена на подготовленных данных. Машинное обучение
(Machine Learning, ML) требует явного определения целевой переменной, входных
признаков, исключенных столбцов и критерия проверки качества.

Некоторые столбцы допустимы для объяснительного анализа, но недопустимы как
входные признаки модели. Например, `output_power_w` полезен для проверки
энергетического баланса, но при прогнозе КПД может привести к утечке данных,
поскольку КПД расчетно связан с мощностью.
"""),
        code("""
ml_task_spec = pd.DataFrame(
    [
        {
            "task_type": "регрессия (regression)",
            "target": "efficiency",
            "allowed_features": "speed_rpm, torque_nm, voltage_v, temperature_c, ambient_temp_c",
            "excluded_columns": "sample_id, profile_id, current_a, winding_resistance_ohm, output_power_w, loss_power_w, torque_limit_nm",
            "reason": "служебные, расчетные и физически тождественно связанные столбцы не должны раскрывать целевую переменную",
        },
        {
            "task_type": "классификация (classification)",
            "target": "производная метка допустимости режима",
            "allowed_features": "первичные измеряемые признаки и заранее разрешенные расчетные признаки",
            "excluded_columns": "mode_label, violation_count, признаки причин нарушений, запасы до ограничений",
            "reason": "модель не должна получать прямое описание правила разметки",
        },
    ]
)

ml_task_spec
"""),
        code("""
data_quality_checklist = pd.Series(
    {
        "no_missing_after_cleaning": int(clean_df.isna().sum().sum()) == 0,
        "sample_id_unique": clean_df["sample_id"].is_unique,
        "efficiency_between_0_and_1": clean_df["efficiency"].between(0, 1).all(),
        "nonnegative_power": (clean_df["output_power_w"] >= 0).all(),
        "feature_columns_defined": len(feature_columns) > 0,
        "target_column_defined": target_column in clean_df.columns,
    },
    name="passed",
)

data_quality_checklist.to_frame()
"""),
        md("""
## Задание для аудиторного отчета

1. Укажите, какие столбцы являются признаками, а какой столбец является
   целевой переменной.
2. Приведите таблицу пропусков.
3. Вставьте два графика: КПД от момента и температуру от тока.
4. Опишите не менее одного возможного выброса и укажите, почему его нельзя
   удалять без инженерного объяснения.
5. Проверьте энергетический баланс по формулам из теоретического блока и
   укажите максимальную относительную ошибку пересчета `output_power_w`.
6. Сформулируйте индивидуальный вывод каждого участника группы: какой признак
   наиболее важен для первичного анализа КПД и почему.
7. В отчете явно разделите измеряемые, расчетные и служебные столбцы.

Ответ группы:

"""),
        *dataset_assignment_cells(lesson_number=1),
    ]

    if teacher:
        cells.extend(
            [
                md("""
## Методический комментарий для преподавателя

Ориентиры для проверки:

1. таблица содержит 240 наблюдений и 13 столбцов;
2. в столбцах `torque_nm`, `current_a`, `temperature_c`, `efficiency`
   должно быть по 6 пропусков;
3. возможные выбросы должны обсуждаться как кандидаты, а не автоматически
   удаляться;
4. корректный вывод должен связывать КПД с режимом работы, а температуру -
   с током и потерями.
"""),
                code("""
print("Контрольная проверка числа пропусков")
print(df[["torque_nm", "current_a", "temperature_c", "efficiency"]].isna().sum())

print("\\nКорреляции с КПД после медианной замены")
print(clean_df[feature_columns + [target_column]].corr(numeric_only=True)[target_column].sort_values(ascending=False))
"""),
            ]
        )

    return cells


def notebook_02(teacher: bool) -> list[nbf.NotebookNode]:
    title_suffix = " Версия преподавателя" if teacher else ""
    regression_feature_cell = (
        f"""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


feature_columns = {REGRESSION_STRICT_FEATURES!r}
target_column = "efficiency"

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
"""
        if teacher
        else """
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# TODO: заполните строгий набор признаков без `current_a`, `output_power_w`
# и `loss_power_w`. Используйте только столбцы feature-CSV.
feature_columns = [
    # "speed_rpm",
    # "torque_nm",
    # "voltage_v",
    # "temperature_c",
    # "ambient_temp_c",
]
target_column = "efficiency"

if not feature_columns:
    raise ValueError("Заполните `feature_columns` перед обучением модели.")

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
"""
    )
    regression_experiment_cell = (
        """
# Самостоятельный эксперимент.
# Измените значения degree и alpha, затем повторно выполните ячейку.
experiment_degree = 2
experiment_alpha = 1.0

experiment_model = Pipeline(
    steps=[
        ("input_scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures(degree=experiment_degree, include_bias=False)),
        ("polynomial_scaler", StandardScaler()),
        ("model", Ridge(alpha=experiment_alpha)),
    ]
)

experiment_model.fit(X_train, y_train)
experiment_pred = experiment_model.predict(X_test)
pd.Series(regression_metrics(y_test, experiment_pred), name="experiment")
"""
        if teacher
        else """
# TODO: задайте параметры самостоятельного эксперимента.
# `experiment_degree` - степень полиномиальных признаков.
# `experiment_alpha` - коэффициент L2-регуляризации гребневой регрессии.
experiment_degree = None
experiment_alpha = None

if experiment_degree is None or experiment_alpha is None:
    raise ValueError("Задайте `experiment_degree` и `experiment_alpha`.")

experiment_model = Pipeline(
    steps=[
        ("input_scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures(degree=experiment_degree, include_bias=False)),
        ("polynomial_scaler", StandardScaler()),
        ("model", Ridge(alpha=experiment_alpha)),
    ]
)

experiment_model.fit(X_train, y_train)
experiment_pred = experiment_model.predict(X_test)
pd.Series(regression_metrics(y_test, experiment_pred), name="experiment")
"""
    )
    cells: list[nbf.NotebookNode] = [
        md(f"""
# Практическое занятие 2. Линейная и полиномиальная регрессия для оценки КПД{title_suffix}

## Назначение занятия

Цель занятия - построить регрессионную модель для оценки коэффициента
полезного действия электромеханического преобразователя.

Регрессия (regression) - задача машинного обучения, в которой требуется
предсказать непрерывную числовую величину. В данном занятии целевая переменная
- `efficiency`, то есть КПД в долях единицы.
"""),
        md("""
## Теоретические сведения

Линейная регрессия (linear regression) строит зависимость вида
$\\hat{y} = b_0 + b_1 x_1 + ... + b_p x_p$, где $\\hat{y}$ - прогноз модели,
$x_1...x_p$ - признаки, а $b_0...b_p$ - настраиваемые параметры.

Полиномиальная регрессия (polynomial regression) добавляет к исходным
признакам степени и произведения признаков. Это позволяет описывать нелинейные
зависимости, но увеличивает риск переобучения.

Регуляризация (regularization) - ограничение сложности модели. В гребневой
регрессии (Ridge regression) используется L2-регуляризация, то есть штраф за
большие значения коэффициентов модели:

$J(b) = \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 + \\alpha \\sum_{j=1}^{p} b_j^2$.

Параметр $\\alpha$ задает силу штрафа: чем он больше, тем сильнее модель
ограничивает абсолютные значения коэффициентов.

В этом занятии регрессионная модель сравнивается с базовой моделью
(baseline model). Базовая модель не использует признаки и всегда прогнозирует
среднее значение КПД по обучающей выборке. Если более сложная модель не
превосходит такой ориентир, ее применение методически не обосновано.
"""),
        md("""
## Как работает линейная регрессионная модель

Линейная регрессия (linear regression) рассматривает прогноз как взвешенную
сумму признаков. В матричной форме модель записывается так:

$$\\hat{y} = Xb,$$

где `X` - матрица признаков, `b` - вектор коэффициентов, `y` - вектор
фактических значений целевой переменной, `y_hat` - вектор прогнозов. Если
используется свободный член, к матрице `X` добавляется столбец единиц.

Обучение модели означает подбор таких коэффициентов, при которых прогнозы
минимально отличаются от фактических значений на обучающей выборке. В обычной
линейной регрессии минимизируется сумма квадратов ошибок:

$$L(b) = \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2.$$

Ошибка `e_i = y_i - y_hat_i` называется остатком (residual). Остаток
показывает, какая часть фактического значения не объяснена моделью. После
обучения важно анализировать не только среднюю ошибку, но и структуру
остатков. Если остатки систематически зависят от скорости, тока или
температуры, модель не описывает часть физической зависимости.

Коэффициент линейной модели показывает направление связи между признаком и
прогнозом при прочих равных условиях. Однако такая интерпретация корректна
только с учетом масштаба признаков и корреляций между ними. Поэтому в
блокноте используется стандартизация: из каждого признака вычитается среднее
значение и результат делится на стандартное отклонение. Стандартизованный
признак имеет среднее значение около 0 и стандартное отклонение около 1.
"""),
        md("""
## Как работает полиномиальная гребневая регрессия

Полиномиальные признаки позволяют линейной модели описывать нелинейные
зависимости. Например, если исходными признаками являются `x1` и `x2`, то при
степени 2 к ним могут быть добавлены `x1^2`, `x1*x2` и `x2^2`. После такого
преобразования модель остается линейной по коэффициентам, но становится
нелинейной по исходным физическим признакам:

$$\\hat{y} = b_0 + b_1x_1 + b_2x_2 + b_3x_1^2 + b_4x_1x_2 + b_5x_2^2.$$

Увеличение числа признаков повышает гибкость модели, но одновременно
увеличивает риск переобучения. Переобучение (overfitting) - ситуация, когда
модель слишком точно описывает обучающую выборку и хуже переносится на новые
данные.

Гребневая регрессия (Ridge regression) уменьшает этот риск с помощью
L2-регуляризации. Регуляризация добавляет штраф за большие коэффициенты:

$$J(b) = \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 +
\\alpha \\sum_{j=1}^{p} b_j^2.$$

Параметр `alpha` управляет силой штрафа. При малом `alpha` модель ближе к
обычной линейной регрессии. При большом `alpha` коэффициенты сильнее
сжимаются к нулю, модель становится устойчивее, но может потерять важные
зависимости. В инженерной интерпретации это означает компромисс между
гибкостью модели и устойчивостью прогноза.
"""),
        md(source_section()),
        md(datasets_for_lessons_01_02_section()),
        code(common_setup_code(
            "practice_02_motor_efficiency_features.csv",
            "practice_02_motor_efficiency_diagnostics.csv",
        )),
        md("""
## Логическая схема регрессионного моделирования

Регрессионная модель должна проверяться на данных, которые не использовались
при обучении. Поэтому процедура включает явное разделение на обучающую и
тестовую выборки. Масштабирование признаков выполняется только внутри
вычислительного конвейера (pipeline), чтобы одна и та же процедура применялась
к обучающим и тестовым данным.
"""),
        md("""
## Последовательность подготовки данных и использования регрессионной модели

Регрессионный эксперимент должен выполняться в фиксированной
последовательности. Нарушение порядка может привести к утечке данных и
завышенной оценке качества.

1. Сформулировать задачу: указать, какая непрерывная величина прогнозируется,
   в каких единицах она измеряется и зачем такой прогноз нужен инженерно.
2. Выбрать целевую переменную (target variable): в данном занятии это
   `efficiency`, то есть коэффициент полезного действия.
3. Определить момент прогноза: какие признаки доступны до расчета КПД, а
   какие появляются только после расчета мощности и потерь.
4. Исключить признаки с утечкой данных (data leakage): целевую переменную,
   производные от нее величины и служебные поля, которые прямо раскрывают
   ответ.
5. Проверить распределение целевой переменной и признаков: слишком узкий
   диапазон цели ограничивает применимость модели.
6. Разделить данные на обучающую и тестовую выборки (train/test split).
   Обучающая выборка используется для подбора параметров, тестовая - только
   для независимой проверки.
7. Построить базовую модель (baseline model). В регрессии таким ориентиром
   может быть прогноз средним значением целевой переменной.
8. Создать вычислительный конвейер (pipeline), в котором предобработка и
   модель объединены в одну процедуру. Это предотвращает ситуацию, когда
   параметры масштабирования вычислены по всей таблице до разбиения.
9. Обучить модель только на обучающей выборке.
10. Получить прогнозы на тестовой выборке и рассчитать MAE, RMSE и R2.
11. Построить график фактических и прогнозных значений, а также график
    остатков. Эти графики помогают увидеть систематические ошибки.
12. Сравнить модель с базовым ориентиром и сформулировать инженерное
    ограничение применения модели.

После обучения модель используется следующим образом: новое наблюдение
проходит те же операции предобработки, затем модель вычисляет прогноз
`y_hat`. Если новое наблюдение выходит за диапазоны обучающих данных, прогноз
следует считать экстраполяцией, то есть применением модели вне области, где
она была проверена. Экстраполяция требует отдельной инженерной осторожности.
"""),
        code("""
draw_process_diagram(
    [
        "Данные",
        "Признаки и цель",
        "Train/Test",
        "Модель",
        "Метрики",
        "Остатки",
        "Вывод",
    ],
    "Схема регрессионного эксперимента",
)
"""),
        code("""
df = pd.read_csv(DATA_FILE)
diagnostics_df = pd.read_csv(DIAGNOSTICS_FILE)
full_df = df.merge(diagnostics_df, on="sample_id", how="left", validate="one_to_one")

print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")
df.head()
"""),
        md("""
## Структура регрессионного набора данных

В занятии 2 используется таблица без искусственно добавленных пропусков.
Каждая строка соответствует одному режиму электромеханического
преобразователя. Целевая переменная `efficiency` является непрерывной
величиной, поэтому задача относится к регрессии. Перед обучением модели
необходимо проверить распределение целевой переменной и связь цели с
основными физическими признаками.
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["efficiency"], bins=24, color="#4c78a8", edgecolor="white")
axes[0].set_title("Распределение КПД")
axes[0].set_xlabel("КПД, доли единицы")
axes[0].set_ylabel("Число наблюдений")

axes[1].scatter(df["speed_rpm"], df["efficiency"], alpha=0.65)
axes[1].set_title("КПД и скорость")
axes[1].set_xlabel("Скорость, rpm")
axes[1].set_ylabel("КПД")

axes[2].scatter(full_df["current_a"], full_df["efficiency"], alpha=0.65, color="tab:orange")
axes[2].set_title("КПД и ток")
axes[2].set_xlabel("Ток, A")
axes[2].set_ylabel("КПД")

plt.tight_layout()
plt.show()
"""),
        md("""
## Выбор признаков

В признаки включаются только величины, которые доступны до расчета КПД:
`speed_rpm`, `torque_nm`, `voltage_v`, `temperature_c`,
`ambient_temp_c`. Столбцы `current_a`, `output_power_w`, `loss_power_w`,
`torque_limit_nm` и сама целевая переменная `efficiency` не используются в
строгой базовой постановке как входные признаки, поскольку ток совместно со
скоростью, моментом и напряжением позволяет косвенно восстановить КПД через
энергетическое тождество, а мощность и потери раскрывают цель напрямую.

Утечка данных (data leakage) - ситуация, когда в признаки попадает информация,
которая в реальной задаче недоступна на момент прогноза или прямо связана с
целевой переменной.
"""),
        md("""
## Критерии допустимости признаков

Критерии допустимости признаков (feature eligibility criteria) определяют,
можно ли использовать столбец как вход модели. Признак допустим, если он
доступен в момент предполагаемого прогноза, имеет понятный физический смысл и
не является прямым преобразованием целевой переменной.

Следует различать три группы столбцов:

1. измеряемые признаки (measured features): ток, напряжение, скорость, момент,
   температура;
2. расчетные признаки (derived features): мощность, потери, КПД, запасы до
   ограничений;
3. служебные признаки (technical identifiers): идентификаторы строк и профилей.

Измеряемые признаки обычно допустимы при физическом обосновании. Расчетные
признаки требуют проверки на утечку данных. Служебные признаки не должны
использоваться как физические входы модели.
"""),
        code(regression_feature_cell),
        md("""
## Диагностика разбиения на обучающую и тестовую выборки

После разбиения необходимо проверить, что обучающая и тестовая выборки
сопоставимы по распределению целевой переменной. Если тестовая выборка
содержит только узкий диапазон КПД, итоговые метрики будут плохо отражать
качество модели на всем диапазоне режимов.

Для реальных временных рядов случайное разбиение может быть недостаточным:
соседние точки одного профиля измерения похожи друг на друга, поэтому модель
может получить завышенную оценку качества. В таких случаях применяют разбиение
по времени или по группам профилей.
"""),
        code("""
split_diagnostics = pd.DataFrame(
    {
        "train": y_train.describe(),
        "test": y_test.describe(),
    }
)
split_diagnostics
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(y_train, bins=20, alpha=0.65, label="обучающая выборка", color="#4c78a8")
ax.hist(y_test, bins=20, alpha=0.65, label="тестовая выборка", color="#f58518")
ax.set_xlabel("КПД, доли единицы")
ax.set_ylabel("Число наблюдений")
ax.set_title("Сравнение распределений целевой переменной после разбиения")
ax.legend()
plt.tight_layout()
plt.show()
"""),
        code("""
plot_correlation_heatmap(
    df,
    feature_columns + [target_column],
    "Корреляции признаков и целевой переменной в регрессионной задаче",
)
"""),
        md("""
## Обучение моделей

Обучающая выборка используется для настройки параметров модели. Тестовая
выборка используется только для итоговой проверки качества на данных, которые
модель не видела при обучении.
"""),
        md("""
## Вычислительный конвейер модели

Вычислительный конвейер модели (model pipeline) объединяет предварительную
обработку признаков и саму модель в одну процедуру. Это важно, потому что все
параметры предварительной обработки должны оцениваться только на обучающей
выборке. В данном занятии стандартизация (standardization), построение
полиномиальных признаков (polynomial features) и регрессия выполняются как
последовательные этапы одного конвейера.

Такой подход уменьшает риск методической ошибки: модель обучается и
применяется к новым данным с одной и той же последовательностью преобразований.
"""),
        code("""
linear_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)

polynomial_ridge_model = Pipeline(
    steps=[
        ("input_scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("polynomial_scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]
)

models = {
    "linear_regression": linear_model,
    "polynomial_ridge": polynomial_ridge_model,
}

for model in models.values():
    model.fit(X_train, y_train)
"""),
        md("""
## Интерпретация коэффициентов линейной модели

После стандартизации признаков коэффициенты линейной модели можно использовать
как приблизительный ориентир направления влияния признаков. Положительный
коэффициент означает, что при прочих равных условиях рост стандартизованного
признака связан с ростом прогноза КПД. Отрицательный коэффициент связан со
снижением прогноза. Такой вывод является статистическим и не заменяет
физическое обоснование.
"""),
        code("""
linear_coefficients = pd.Series(
    linear_model.named_steps["model"].coef_,
    index=feature_columns,
).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
linear_coefficients.plot(kind="barh", ax=ax, color="#4c78a8")
ax.axvline(0.0, color="black", linewidth=1)
ax.set_xlabel("Коэффициент при стандартизованном признаке")
ax.set_title("Коэффициенты линейной регрессии")
plt.tight_layout()
plt.show()

linear_coefficients.to_frame("coefficient")
"""),
        md("""
## Метрики качества

Средняя абсолютная ошибка (Mean Absolute Error, MAE) показывает средний модуль
ошибки. Корень из средней квадратичной ошибки (Root Mean Squared Error, RMSE)
сильнее штрафует крупные ошибки. Коэффициент детерминации (coefficient of
determination, R2) показывает, какая доля изменчивости целевой переменной
объясняется моделью относительно прогноза средним значением.

$MAE = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$.

$RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$.

$R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i - \\bar{y})^2}$.

MAE и RMSE выражаются в единицах целевой переменной. В данной работе это доли
единицы КПД; после умножения на 100 они интерпретируются как процентные
пункты КПД. Коэффициент детерминации $R^2$ является безразмерной метрикой и
на тестовой выборке может быть отрицательным, если модель хуже прогноза
средним значением.
"""),
        md("""
## Правило интерпретации качества модели

Качество регрессионной модели оценивается не по одной метрике, а по
совокупности признаков: улучшение относительно базовой модели, величина
ошибки в физических единицах, отсутствие выраженной структуры остатков и
устойчивость результата при изменении сложности модели. Высокое значение R2
не является самостоятельным доказательством физической корректности модели.

Модель допустима для предварительной инженерной оценки только в пределах
области данных, на которой она проверялась. Применение за пределами диапазона
скоростей, моментов, токов и температур является экстраполяцией
(extrapolation) и требует отдельного обоснования.
"""),
        code("""
def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


metrics = []
predictions = {}

baseline_pred = np.full(shape=len(y_test), fill_value=y_train.mean(), dtype=float)
predictions["mean_baseline"] = baseline_pred
baseline_row = {"model": "mean_baseline"}
baseline_row.update(regression_metrics(y_test, baseline_pred))
metrics.append(baseline_row)

for name, model in models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    row = {"model": name}
    row.update(regression_metrics(y_test, y_pred))
    metrics.append(row)

metrics_df = pd.DataFrame(metrics).set_index("model")
metrics_df
"""),
        md("""
## Улучшение относительно базовой модели

MAE и RMSE выражены в долях единицы КПД. Умножение ошибки на 100 переводит ее
в процентные пункты КПД. Например, MAE = 0.02 означает среднюю абсолютную
ошибку около двух процентных пунктов.
"""),
        code("""
metrics_interpretation = metrics_df.copy()
metrics_interpretation["MAE_percent_points"] = metrics_interpretation["MAE"] * 100.0
metrics_interpretation["RMSE_percent_points"] = metrics_interpretation["RMSE"] * 100.0
metrics_interpretation["MAE_improvement_vs_baseline"] = (
    metrics_interpretation.loc["mean_baseline", "MAE"] - metrics_interpretation["MAE"]
) * 100.0
metrics_interpretation["RMSE_improvement_vs_baseline"] = (
    metrics_interpretation.loc["mean_baseline", "RMSE"] - metrics_interpretation["RMSE"]
) * 100.0

metrics_interpretation
"""),
        md("""
## Демонстрация косвенной и прямой утечки данных

Косвенная утечка данных возникает, когда целевая переменная не включена в
признаки напрямую, но может быть восстановлена из их нелинейной комбинации.
Для КПД действует энергетическое соотношение
`efficiency = output_power_w / (voltage_v * current_a)`, а
`output_power_w = torque_nm * omega`. Поэтому одновременное использование
`speed_rpm`, `torque_nm`, `voltage_v` и `current_a` позволяет полиномиальной
модели приблизиться к расчетной формуле КПД.

Прямая утечка данных возникает при включении `output_power_w` или
`loss_power_w`: эти величины являются расчетными компонентами целевой
переменной. Ниже такой вариант оставлен только как контролируемая
демонстрация методической ошибки.
"""),
        code("""
leakage_demo_columns = [
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "temperature_c",
    "ambient_temp_c",
    "current_a",
    "output_power_w",
    "loss_power_w",
]

leakage_X = full_df[leakage_demo_columns]
leakage_X_train = leakage_X.loc[X_train.index]
leakage_X_test = leakage_X.loc[X_test.index]

leakage_demo_model = Pipeline(
    steps=[
        ("input_scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("polynomial_scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]
)
leakage_demo_model.fit(leakage_X_train, y_train)
leakage_demo_pred = leakage_demo_model.predict(leakage_X_test)

leakage_metrics = pd.DataFrame(
    [
        {"model": "strict_polynomial_ridge", **regression_metrics(y_test, predictions["polynomial_ridge"])},
        {"model": "leakage_demo_polynomial_ridge", **regression_metrics(y_test, leakage_demo_pred)},
    ]
).set_index("model")

leakage_metrics
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    ax.bar(leakage_metrics.index, leakage_metrics[metric], color=["#4c78a8", "#e45756"])
    ax.set_title(metric)
    ax.set_ylabel("Значение")
    ax.tick_params(axis="x", rotation=20)

plt.suptitle("Строгая постановка и демонстрация утечки данных", y=1.05)
plt.tight_layout()
plt.show()
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    axes_data = metrics_df[metric]
    colors = ["#8c8c8c", "#4c78a8", "#f58518"][: len(axes_data)]
    ax.bar(axes_data.index, axes_data.values, color=colors)
    ax.set_title(metric)
    ax.set_ylabel("Значение")
    ax.tick_params(axis="x", rotation=20)

plt.suptitle("Сравнение качества регрессионных моделей", y=1.05)
plt.tight_layout()
plt.show()
"""),
        md("""
## Графики прогноза и остатков

Остаток - разность между фактическим значением и прогнозом модели. Если
остатки имеют выраженную структуру, модель не описывает часть зависимости.
"""),
        code("""
selected_model_name = "polynomial_ridge"
selected_pred = predictions[selected_model_name]
residuals = y_test - selected_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, selected_pred, alpha=0.75)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black")
axes[0].set_xlabel("Фактический КПД")
axes[0].set_ylabel("Прогноз КПД")
axes[0].set_title("Прогноз и фактические значения")

axes[1].scatter(selected_pred, residuals, alpha=0.75, color="tab:orange")
axes[1].axhline(0.0, color="black")
axes[1].set_xlabel("Прогноз КПД")
axes[1].set_ylabel("Остаток")
axes[1].set_title("График остатков")

plt.tight_layout()
plt.show()
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=24, color="#54a24b", edgecolor="white")
ax.axvline(0.0, color="black", linewidth=1)
ax.set_xlabel("Остаток модели")
ax.set_ylabel("Число наблюдений")
ax.set_title("Распределение остатков полиномиальной гребневой регрессии")
plt.tight_layout()
plt.show()
"""),
        md("""
## Ошибки по диапазонам КПД

Средняя метрика может скрывать то, что модель хорошо работает в одной области
КПД и хуже в другой. Поэтому полезно сгруппировать тестовые наблюдения по
диапазонам фактического КПД и построить диаграмму размаха остатков.
"""),
        code("""
residual_analysis = pd.DataFrame(
    {
        "actual_efficiency": y_test,
        "predicted_efficiency": selected_pred,
        "residual": residuals,
    }
).reset_index(drop=True)
residual_analysis["efficiency_band"] = pd.cut(
    residual_analysis["actual_efficiency"],
    bins=[0.0, 0.55, 0.65, 0.75, 0.85, 1.0],
    include_lowest=True,
)

bands = residual_analysis["efficiency_band"].cat.categories
box_data = [
    residual_analysis.loc[residual_analysis["efficiency_band"] == band, "residual"]
    for band in bands
]

fig, ax = plt.subplots(figsize=(10, 4))
ax.boxplot(box_data, labels=[str(band) for band in bands])
ax.axhline(0.0, color="black", linewidth=1)
ax.set_xlabel("Диапазон фактического КПД")
ax.set_ylabel("Остаток")
ax.set_title("Остатки модели по диапазонам КПД")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()

residual_analysis.groupby("efficiency_band", observed=True)["residual"].agg(["count", "mean", "std"])
"""),
        md("""
## Влияние сложности модели

Сложность модели должна соответствовать сложности физической зависимости. Если
модель слишком проста, она не описывает существенную нелинейность. Если модель
слишком сложна, она может начать описывать случайный шум обучающей выборки.
Это явление называется переобучением, overfitting.
"""),
        code("""
complexity_rows = []

for degree in [1, 2, 3, 4]:
    model = Pipeline(
        steps=[
            ("input_scaler", StandardScaler()),
            ("polynomial_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("polynomial_scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    complexity_rows.append(
        {
            "degree": degree,
            "train_R2": r2_score(y_train, model.predict(X_train)),
            "test_R2": r2_score(y_test, model.predict(X_test)),
        }
    )

complexity_df = pd.DataFrame(complexity_rows)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(complexity_df["degree"], complexity_df["train_R2"], marker="o", label="Обучающая выборка")
ax.plot(complexity_df["degree"], complexity_df["test_R2"], marker="o", label="Тестовая выборка")
ax.set_xlabel("Степень полиномиальных признаков")
ax.set_ylabel("R2")
ax.set_title("Связь сложности модели и качества прогноза")
ax.legend()
plt.tight_layout()
plt.show()

complexity_df
"""),
        md("""
## Задание для аудиторного отчета

1. Укажите признаки и целевую переменную.
2. Сравните базовую модель, линейную регрессию и полиномиальную гребневую
   регрессию (Ridge regression) по MAE, RMSE и R2.
3. Опишите, какая модель лучше описывает тестовую выборку.
4. Переведите MAE и RMSE лучшей модели в процентные пункты КПД.
5. Измените степень полинома или параметр `alpha` и укажите, как меняются
   метрики качества на обучающей и тестовой выборках.
6. Объясните, почему нельзя использовать `output_power_w` и `loss_power_w`
   без проверки на утечку данных.
7. Укажите, можно ли использовать модель для предварительной инженерной
   оценки КПД, и назовите не менее двух ограничений такого применения.

Ответ группы:

"""),
        code(regression_experiment_cell),
        *dataset_assignment_cells(lesson_number=2),
    ]

    if teacher:
        cells.extend(
            [
                md("""
## Методический комментарий для преподавателя

При фиксированном разбиении данных ожидаемый порядок результатов:

1. базовая модель среднего значения имеет MAE около 0.093 и R2 около 0;
2. линейная регрессия на строгих признаках дает R2 около 0.58;
3. полиномиальная гребневая регрессия второй степени дает R2 около 0.75;
4. демонстрация утечки с `current_a`, `output_power_w` и `loss_power_w`
   может дать R2 около 0.94, но этот результат не является допустимым
   ориентиром качества базовой модели;
5. вывод должен подчеркивать, что более высокая метрика не отменяет
   необходимости физической проверки модели;
6. студент должен заметить риск прямой и косвенной утечки данных.
"""),
                code("""
print(metrics_df.round(4))
"""),
            ]
        )

    return cells


def notebook_03(teacher: bool) -> list[nbf.NotebookNode]:
    title_suffix = " Версия преподавателя" if teacher else ""
    classification_feature_cell = (
        f"""
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


feature_columns = {CLASSIFICATION_STRICT_FEATURES!r}
target_column = "is_allowed"

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
"""
        if teacher
        else """
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# TODO: заполните строгий набор признаков. Не включайте `efficiency`,
# `output_power_w`, `mode_label`, `violation_count` и столбцы `*_margin`.
feature_columns = [
    # "speed_rpm",
    # "torque_nm",
    # "voltage_v",
    # "current_a",
    # "ambient_temp_c",
    # "temperature_c",
]
target_column = "is_allowed"

if not feature_columns:
    raise ValueError("Заполните `feature_columns` перед обучением дерева решений.")

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
"""
    )
    classification_experiment_cell = (
        """
# Самостоятельный эксперимент.
# Измените max_depth и сравните результаты с базовой моделью.
experiment_depth = 3

experiment_tree = DecisionTreeClassifier(
    max_depth=experiment_depth,
    min_samples_leaf=8,
    random_state=RANDOM_STATE,
)
experiment_tree.fit(X_train, y_train)
experiment_pred = experiment_tree.predict(X_test)

pd.Series(
    classification_metrics_dict(y_test, experiment_pred),
    name="experiment",
)
"""
        if teacher
        else """
# TODO: задайте максимальную глубину дерева решений.
# `experiment_depth` управляет числом последовательных условий от корня к листу.
experiment_depth = None

if experiment_depth is None:
    raise ValueError("Задайте `experiment_depth` перед запуском эксперимента.")

experiment_tree = DecisionTreeClassifier(
    max_depth=experiment_depth,
    min_samples_leaf=8,
    random_state=RANDOM_STATE,
)
experiment_tree.fit(X_train, y_train)
experiment_pred = experiment_tree.predict(X_test)

pd.Series(
    classification_metrics_dict(y_test, experiment_pred),
    name="experiment",
)
"""
    )
    cells: list[nbf.NotebookNode] = [
        md(f"""
# Практическое занятие 3. Дерево решений для классификации режимов электропривода{title_suffix}

## Назначение занятия

Цель занятия - построить интерпретируемую модель, которая аппроксимирует
учебное инженерное правило допустимости режима электропривода.

Классификация (classification) - задача машинного обучения, в которой модель
относит наблюдение к одному из заранее заданных классов. В данной работе
целевая переменная `is_allowed` принимает два значения: 1 - режим допустим,
0 - режим недопустим. Важно понимать, что классы сформированы по заданным
учебным порогам тока, температуры, скорости, КПД и момента; дерево решений
учится воспроизводить это инженерное правило по измеряемым признакам.
"""),
        md("""
## Теоретические сведения

Дерево решений (Decision Tree) последовательно разбивает пространство признаков
на области с помощью условий вида `feature <= threshold`, где `feature` -
признак, а `threshold` - порог. Узел (node) содержит условие разбиения, лист
(leaf) содержит итоговое решение модели, глубина (depth) показывает число
условий на пути от корня дерева до листа. Преимущество дерева решений -
интерпретируемость: результат можно представить как набор правил.

Внутри дерева используется критерий неоднородности узла. В данном занятии
применяется индекс Джини (Gini impurity). Для узла с долями классов `p_k`
он записывается так:

```text
Gini = 1 - sum(p_k^2)
```

Если в узле находятся объекты только одного класса, индекс Джини равен 0.
Если классы перемешаны, значение индекса больше. Обучение дерева выбирает
разделения, которые сильнее уменьшают неоднородность дочерних узлов.

Переобучение (overfitting) - ситуация, когда модель слишком точно подстроилась
под обучающую выборку и хуже работает на новых данных. Для дерева решений
риск переобучения уменьшают ограничением глубины `max_depth` и минимального
числа наблюдений в листе `min_samples_leaf`.
"""),
        md("""
## Как дерево решений строит классификационную модель

Дерево решений является жадным алгоритмом (greedy algorithm). Это означает,
что на каждом шаге модель выбирает локально лучшее разбиение текущего узла,
не перебирая все возможные деревья целиком. Для каждого признака и каждого
кандидатного порога проверяется условие вида

```text
feature <= threshold
```

После такого условия наблюдения делятся на левую и правую дочерние группы.
Качество разбиения оценивается по уменьшению неоднородности. Если используется
индекс Джини, взвешенная неоднородность после разбиения равна

$$G_{split} =
\\frac{n_L}{n}G_L + \\frac{n_R}{n}G_R,$$

где `n_L` и `n_R` - числа наблюдений в левом и правом дочерних узлах,
`G_L` и `G_R` - индексы Джини в этих узлах. Модель выбирает разбиение, при
котором значение `G_split` минимально, то есть классы в дочерних узлах
становятся более однородными.

В листе дерева прогнозируется класс большинства среди обучающих наблюдений,
попавших в этот лист. Для нового наблюдения модель последовательно проверяет
условия от корня до листа. Этот путь называется путем решения (decision path).
Именно поэтому дерево решений можно интерпретировать как набор правил:

```text
если условие 1 выполнено и условие 2 выполнено, то прогнозируется класс A
```

Вероятность класса в листе обычно оценивается как доля объектов этого класса
среди обучающих наблюдений листа. Например, если в листе 18 допустимых и
2 недопустимых режима, оценка вероятности допустимого режима равна 18 / 20.
Такая вероятность является эмпирической, то есть рассчитанной по данным, и
не должна трактоваться как нормативная вероятность безопасности оборудования.

Ограничения дерева решений:

1. дерево строит прямоугольные области в пространстве признаков, поэтому
   гладкие физические зависимости описываются ступенчато;
2. небольшие изменения данных могут изменить структуру дерева;
3. слишком глубокое дерево склонно запоминать частные особенности выборки;
4. важность признаков показывает вклад в разбиения, но не доказывает
   физическую причинность.
"""),
        md(source_section()),
        md(datasets_for_lesson_03_section()),
        code(common_setup_code(
            "practice_03_drive_mode_features.csv",
            "practice_03_drive_mode_diagnostics.csv",
        )),
        md("""
## Логическая схема классификации режимов

В классификационной задаче сначала определяется инженерное правило, по
которому режим считается допустимым или недопустимым. Затем это правило
превращается в целевую переменную. Модель не должна получать в качестве
признаков прямое описание этого правила, иначе возникнет утечка данных.
"""),
        md("""
## Последовательность подготовки данных и использования классификационной модели

Классификация режимов отличается от регрессии тем, что модель прогнозирует не
числовую величину, а класс. Класс - заранее заданная категория состояния
объекта. В данном занятии классы имеют инженерный смысл: допустимый и
недопустимый режим электропривода.

Рекомендуемая последовательность действий:

1. Определить инженерные ограничения: предельный ток, предельную температуру,
   предельную скорость, минимальный КПД и предельный момент.
2. Сформировать целевую переменную `is_allowed`: `1` - допустимый режим,
   `0` - недопустимый режим. Правило формирования класса должно быть
   записано явно.
3. Проверить распределение классов. Если один класс встречается редко, доля
   правильных ответов может быть недостаточной метрикой качества.
4. Исключить признаки, прямо раскрывающие ответ: `mode_label`,
   `violation_count`, признаки причин нарушений и запасы до ограничений.
5. Выбрать входные признаки, которые могли бы быть доступны в реальной
   системе мониторинга до расчета диагностических показателей: ток,
   напряжение, скорость, момент и температура.
6. Разделить данные на обучающую и тестовую выборки с сохранением долей
   классов. Такое разбиение называется стратифицированным
   (stratified split).
7. Построить базовую модель большинства (majority baseline), которая всегда
   прогнозирует самый частый класс. Она задает минимальный ориентир качества.
8. Обучить дерево решений с ограниченной глубиной. Ограничение глубины
   повышает интерпретируемость и уменьшает риск переобучения.
9. Рассчитать метрики отдельно для класса `0` и класса `1`. Для безопасности
   особенно важна полнота обнаружения недопустимых режимов.
10. Построить матрицу ошибок и отдельно посчитать долю ложных разрешений,
    когда истинно недопустимый режим ошибочно признан допустимым.
11. Выписать правила дерева решений в инженерной форме и проверить, не
    противоречат ли они физическому смыслу объекта.
12. Оценить влияние параметра `max_depth`: слишком малая глубина дает
    недообучение, слишком большая глубина может привести к переобучению.

При использовании обученного дерева для нового режима измеренные признаки
последовательно проходят через условия дерева от корня к листу. Итоговый лист
задает прогноз класса. Такой прогноз следует рассматривать как
информационную поддержку инженерного решения, а не как самостоятельное
нормативное разрешение эксплуатации оборудования.
"""),
        code("""
draw_process_diagram(
    [
        "Ограничения",
        "Классы",
        "Признаки",
        "Дерево",
        "Матрица ошибок",
        "Решение",
    ],
    "Схема классификации допустимости режима",
)
"""),
        code("""
df = pd.read_csv(DATA_FILE)
diagnostics_df = pd.read_csv(DIAGNOSTICS_FILE)
full_df = df.merge(diagnostics_df, on="sample_id", how="left", validate="one_to_one")

print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")
df.head()
"""),
        md("""
## Реконструкция целевого правила

Перед обучением классификатора необходимо понять, как сформирована целевая
переменная. В учебном наборе `is_allowed` является результатом инженерного
правила: режим допустим только тогда, когда не нарушены ограничения по току,
температуре, скорости, КПД и моменту. Такая постановка полезна для обучения
интерпретации дерева решений, но она не означает, что модель самостоятельно
открывает новый физический закон.

Реконструкция правила нужна для контроля утечки данных. Если в признаки
включить столбцы, которые прямо являются частями правила, дерево будет
воспроизводить заданную разметку, а не решать независимую прогностическую
задачу.
"""),
        code("""
target_rule_table = pd.DataFrame(
    [
        {
            "condition": "overcurrent",
            "engineering_meaning": "ток выше учебного предела",
            "class_effect": "делает режим недопустимым",
        },
        {
            "condition": "overheating",
            "engineering_meaning": "температура выше учебного предела",
            "class_effect": "делает режим недопустимым",
        },
        {
            "condition": "overspeed",
            "engineering_meaning": "скорость выше учебного предела",
            "class_effect": "делает режим недопустимым",
        },
        {
            "condition": "low_efficiency",
            "engineering_meaning": "КПД ниже учебного предела",
            "class_effect": "делает режим недопустимым",
        },
        {
            "condition": "torque_violation",
            "engineering_meaning": "момент выше учебного ограничения для режима",
            "class_effect": "делает режим недопустимым",
        },
    ]
)

rule_reconstructed_is_allowed = (
    full_df[["overcurrent", "overheating", "overspeed", "low_efficiency", "torque_violation"]]
    .sum(axis=1)
    .eq(0)
    .astype(int)
)

print(
    "Число расхождений между is_allowed и восстановленным правилом:",
    int((full_df["is_allowed"] != rule_reconstructed_is_allowed).sum()),
)
target_rule_table
"""),
        md("""
## Структура классификационного набора данных

В учебном наборе каждая строка соответствует одному режиму работы. Целевая
переменная `is_allowed` задает бинарный класс: 1 - допустимый режим, 0 -
недопустимый режим. Дополнительный столбец `mode_label` содержит приоритетную
инженерную причину ограничения. Если в строке есть несколько нарушений,
показывается только первая причина в принятом порядке приоритетов, а полное
число нарушений хранится в столбце `violation_count`. Столбцы причин
нарушений и запасов до пределов нельзя использовать как входные признаки,
поскольку они напрямую раскрывают целевой класс.
"""),
        code("""
violation_columns = [
    "overheating",
    "overcurrent",
    "overspeed",
    "low_efficiency",
    "torque_violation",
]
violation_labels = {
    "overheating": "перегрев",
    "overcurrent": "превышение тока",
    "overspeed": "превышение скорости",
    "low_efficiency": "низкий КПД",
    "torque_violation": "превышение момента",
}

reason_counts = full_df[violation_columns].sum().sort_values(ascending=False)
primary_mode_counts = full_df["mode_label"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(
    [violation_labels[column] for column in reason_counts.index],
    reason_counts.values,
    color="#4c78a8",
)
axes[0].set_ylabel("Число срабатываний причины")
axes[0].set_title("Все причины недопустимости режима")
axes[0].tick_params(axis="x", rotation=25)

axes[1].bar(primary_mode_counts.index, primary_mode_counts.values, color="#f58518")
axes[1].set_ylabel("Число наблюдений")
axes[1].set_title("Приоритетная причина в mode_label")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.show()

pd.DataFrame(
    {
        "all_reason_count": reason_counts,
        "label_ru": [violation_labels[column] for column in reason_counts.index],
    }
)
"""),
        md("""
## Проверка распределения классов

Перед обучением классификатора необходимо проверить, сколько наблюдений
относится к каждому классу. Сильный дисбаланс классов может привести к
формально высокой точности при плохом обнаружении редкого класса.
"""),
        code("""
class_counts = df["is_allowed"].value_counts().sort_index()
class_counts.rename(index={0: "недопустимый режим", 1: "допустимый режим"})
"""),
        code("""
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["недопустимый", "допустимый"]
ax.bar(labels, class_counts.values, color=["#e45756", "#54a24b"])
ax.set_ylabel("Число наблюдений")
ax.set_title("Распределение классов целевой переменной")

for index, value in enumerate(class_counts.values):
    ax.text(index, value + 3, str(value), ha="center")

plt.tight_layout()
plt.show()
"""),
        md("""
## Выбор признаков и исключение утечки данных

Столбец `mode_label` объясняет причину ограничения режима и не должен
использоваться как входной признак. Столбцы с запасами до ограничений также
исключаются из базовой модели, поскольку они прямо построены из пороговых
правил.
"""),
        md("""
## Строгая прогностическая постановка и демонстрация утечки

Если признак входит в правило формирования целевого класса, его использование
меняет смысл задачи. В базовой модели этого занятия используются только
строгие признаки: скорость, момент, напряжение, ток, температура окружающей
среды и температура объекта. Столбцы `efficiency`, `output_power_w`,
`mode_label`, `violation_count` и запасы до ограничений исключены.

Отдельный раздел ниже показывает демонстрацию утечки данных: при включении
`efficiency` и `output_power_w` дерево решений получает расчетные величины,
близкие к правилу разметки, и качество становится методически завышенным.
Этот результат нельзя считать качеством базовой модели.

В отчете необходимо явно указать выбранную постановку. Для диагностической
постановки допустимо использовать `efficiency`, но нельзя утверждать, что
модель обнаружила независимый физический закон.
"""),
        code("""
feature_audit = pd.DataFrame(
    [
        {
            "column": "speed_rpm",
            "feature_type": "измеряемый признак",
            "baseline_use": "используется в строгой модели",
            "strict_use": "используется",
            "risk": "участвует в правиле через overspeed; допустим только при явном указании диагностической постановки",
        },
        {
            "column": "torque_nm",
            "feature_type": "измеряемый признак",
            "baseline_use": "используется в строгой модели",
            "strict_use": "используется",
            "risk": "участвует в оценке перегрузки по моменту",
        },
        {
            "column": "voltage_v",
            "feature_type": "измеряемый признак",
            "baseline_use": "используется в строгой модели",
            "strict_use": "используется",
            "risk": "прямого раскрытия класса нет",
        },
        {
            "column": "current_a",
            "feature_type": "измеряемый признак",
            "baseline_use": "используется в строгой модели",
            "strict_use": "используется",
            "risk": "участвует в правиле через overcurrent",
        },
        {
            "column": "temperature_c",
            "feature_type": "измеряемый признак",
            "baseline_use": "используется в строгой модели",
            "strict_use": "используется",
            "risk": "участвует в правиле через overheating",
        },
        {
            "column": "output_power_w",
            "feature_type": "расчетный признак",
            "baseline_use": "исключается",
            "strict_use": "исключается",
            "risk": "связан с КПД и энергетическим балансом",
        },
        {
            "column": "efficiency",
            "feature_type": "расчетный признак",
            "baseline_use": "исключается",
            "strict_use": "исключается",
            "risk": "прямо участвует в правиле low_efficiency",
        },
        {
            "column": "mode_label, violation_count, current_margin_a, temperature_margin_c, speed_margin_rpm, overcurrent, overheating, overspeed, low_efficiency, torque_violation",
            "feature_type": "служебные признаки правила",
            "baseline_use": "исключаются",
            "strict_use": "исключаются",
            "risk": "прямо раскрывают целевую переменную",
        },
    ]
)

feature_audit
"""),
        code(classification_feature_cell),
        md("""
## Базовая модель классификации

Перед обучением дерева решений необходимо задать простой ориентир качества.
Базовая модель большинства всегда предсказывает класс, который чаще встречался
в обучающей выборке. Такая модель не использует инженерные признаки, поэтому
дерево решений должно превосходить ее не только по общей точности, но и по
полноте обнаружения критически важных классов.
"""),
        code("""
def classification_metrics_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    cm_local = confusion_matrix(y_true, y_pred, labels=[0, 1])
    not_allowed_total = cm_local[0].sum()
    dangerous_false_allowed_rate = (
        cm_local[0, 1] / not_allowed_total if not_allowed_total > 0 else 0.0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_not_allowed": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_not_allowed": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_not_allowed": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_allowed": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_allowed": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_allowed": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "dangerous_false_allowed_rate": dangerous_false_allowed_rate,
    }


majority_class = int(y_train.mode().iloc[0])
baseline_pred = np.full(shape=len(y_test), fill_value=majority_class, dtype=int)

pd.Series(
    classification_metrics_dict(y_test, baseline_pred),
    name="majority_baseline",
).to_frame("value")
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = df["is_allowed"].map({0: "#e45756", 1: "#54a24b"})

axes[0].scatter(full_df["temperature_c"], full_df["efficiency"], c=colors, alpha=0.70)
axes[0].axvline(74.0, color="black", linestyle="--", label="Учебный предел температуры")
axes[0].axhline(0.45, color="black", linestyle=":", label="Учебный предел КПД")
axes[0].set_xlabel("Температура, deg_C")
axes[0].set_ylabel("КПД")
axes[0].set_title("Классы в пространстве температуры и КПД")
axes[0].legend()

axes[1].scatter(df["speed_rpm"], df["current_a"], c=colors, alpha=0.70)
axes[1].axvline(63000.0, color="black", linestyle="--", label="Учебный предел скорости")
axes[1].axhline(11.0, color="black", linestyle=":", label="Учебный предел тока")
axes[1].set_xlabel("Скорость, rpm")
axes[1].set_ylabel("Ток, A")
axes[1].set_title("Классы в пространстве скорости и тока")
axes[1].legend()

plt.tight_layout()
plt.show()
"""),
        md("""
## Обучение дерева решений

Параметр `max_depth=3` ограничивает число последовательных правил. Это делает
модель проще для интерпретации и снижает риск переобучения.
"""),
        code("""
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=8,
    random_state=RANDOM_STATE,
)

tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
"""),
        md("""
## Демонстрация утечки данных

Ниже строится дополнительное дерево решений с признаками `efficiency` и
`output_power_w`. Эта модель не является базовой: она показывает, как
расчетные величины, близкие к правилу разметки, искусственно завышают метрики.
Такой результат следует интерпретировать как пример утечки данных
(data leakage), а не как доказательство высокой прогностической способности.
"""),
        code("""
leakage_demo_columns = [
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "current_a",
    "ambient_temp_c",
    "temperature_c",
    "efficiency",
    "output_power_w",
]

leakage_X = full_df[leakage_demo_columns]
leakage_X_train = leakage_X.loc[X_train.index]
leakage_X_test = leakage_X.loc[X_test.index]

leakage_tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=8,
    random_state=RANDOM_STATE,
)

leakage_tree_model.fit(leakage_X_train, y_train)
leakage_pred = leakage_tree_model.predict(leakage_X_test)
"""),
        md("""
## Метрики классификации

Доля правильных ответов (accuracy) - доля верно классифицированных наблюдений
среди всех наблюдений. Точность положительных предсказаний (precision)
показывает, какая доля объектов, отнесенных моделью к выбранному
положительному классу, действительно принадлежит этому классу. Полнота
(recall) показывает, какая доля объектов выбранного положительного класса
найдена моделью. F1-мера (F1-score) является гармоническим средним precision
и recall.

Для инженерной безопасности отдельно анализируется класс 0, то есть
недопустимый режим. Наиболее опасная ошибка - ложное разрешение: истинный
класс 0, прогноз 1.

$accuracy = (TP + TN)/(TP + TN + FP + FN)$,

$precision = TP/(TP + FP)$,

$recall = TP/(TP + FN)$,

$F1 = 2 \\cdot precision \\cdot recall/(precision + recall)$.

Здесь TP - истинно положительные решения, TN - истинно отрицательные решения,
FP - ложноположительные решения, FN - ложноотрицательные решения. Смысл этих
обозначений зависит от того, какой класс выбран положительным.
"""),
        md("""
## Приоритет метрик для инженерной безопасности

В задачах контроля режима метрики должны интерпретироваться с учетом
последствий ошибки. Для класса "недопустимый режим" основное значение имеет
полнота (recall), поскольку она показывает, какая доля действительно
недопустимых режимов обнаружена моделью.

Показатель `dangerous_false_allowed_rate` отдельно фиксирует долю случаев,
когда недопустимый режим ошибочно признан допустимым. Если общая доля
правильных ответов высока, но полнота для недопустимого класса низка, модель
не должна рассматриваться как приемлемая для инженерного контроля.
"""),
        code("""
classification_metrics_df = pd.DataFrame(
    [
        {
            "model": "majority_baseline",
            **classification_metrics_dict(y_test, baseline_pred),
        },
        {
            "model": "strict_decision_tree",
            **classification_metrics_dict(y_test, y_pred),
        },
        {
            "model": "leakage_demo_decision_tree",
            **classification_metrics_dict(y_test, leakage_pred),
        },
    ]
).set_index("model")

classification_metrics_df
"""),
        code("""
selected_classification_metrics = [
    "accuracy",
    "recall_not_allowed",
    "f1_not_allowed",
    "dangerous_false_allowed_rate",
]

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

for ax, metric in zip(axes, selected_classification_metrics):
    ax.bar(
        classification_metrics_df.index,
        classification_metrics_df[metric],
        color=["#8c8c8c", "#4c78a8", "#e45756"],
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_title(metric)
    ax.tick_params(axis="x", rotation=20)

plt.suptitle("Строгое дерево решений, базовая модель и демонстрация утечки", y=1.05)
plt.tight_layout()
plt.show()
"""),
        code("""
print(classification_report(
    y_test,
    y_pred,
    target_names=["недопустимый режим", "допустимый режим"],
))
"""),
        md("""
## Матрица ошибок

Матрица ошибок (confusion matrix) показывает число правильных и неправильных
решений по каждому классу. Для инженерной диагностики особенно важно
обсуждать ошибки, при которых недопустимый режим ошибочно признается
допустимым.
"""),
        code("""
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["недопустимый", "допустимый"],
)
disp.plot(values_format="d", cmap="Blues")
plt.title("Матрица ошибок")
plt.show()
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cm_count = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_normalized = confusion_matrix(y_test, y_pred, labels=[0, 1], normalize="true")
display_labels = ["недопустимый", "допустимый"]

ConfusionMatrixDisplay(
    confusion_matrix=cm_count,
    display_labels=display_labels,
).plot(values_format="d", cmap="Blues", ax=axes[0], colorbar=False)
axes[0].set_title("Число наблюдений")

ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=display_labels,
).plot(values_format=".2f", cmap="Blues", ax=axes[1], colorbar=False)
axes[1].set_title("Доли внутри истинного класса")

plt.suptitle("Матрица ошибок в абсолютном и нормированном виде", y=1.05)
plt.tight_layout()
plt.show()
"""),
        code("""
dangerous_error_mask = (y_test == 0) & (y_pred == 1)
dangerous_errors = X_test.loc[dangerous_error_mask].copy()
dangerous_errors["true_class"] = y_test.loc[dangerous_error_mask]
dangerous_errors["predicted_class"] = y_pred[dangerous_error_mask.to_numpy()]
dangerous_indices = y_test.index[dangerous_error_mask]
diagnostic_columns_for_errors = [
    "sample_id",
    "mode_label",
    "overcurrent",
    "overheating",
    "overspeed",
    "low_efficiency",
    "torque_violation",
    "violation_count",
]
dangerous_diagnostics = full_df.loc[dangerous_indices, diagnostic_columns_for_errors]
dangerous_errors = dangerous_diagnostics.join(dangerous_errors)

print(
    "Число опасных ошибок, при которых недопустимый режим признан допустимым:",
    len(dangerous_errors),
)
dangerous_errors.head()
"""),
        md("""
## Информационная полнота признаков

Информационная полнота признакового пространства означает, что выбранные
признаки содержат достаточно сведений для различения всех важных классов. В
строгой модели занятия 3 намеренно исключены `efficiency` и `output_power_w`,
поскольку они прямо раскрывают правило `low_efficiency`. Это защищает модель
от утечки данных, но одновременно ухудшает обнаружение части режимов с низким
КПД.

Следующая таблица показывает, какие причины недопустимости модель пропускает
чаще всего. Если recall для `low_efficiency` мал, это не является ошибкой
библиотеки или дерева решений. Это следствие компромисса между защитой от
утечки данных и разрешающей способностью признакового пространства.
"""),
        code("""
test_error_analysis = full_df.loc[y_test.index, diagnostic_columns_for_errors].copy()
test_error_analysis["true_class"] = y_test
test_error_analysis["predicted_class"] = y_pred

not_allowed_by_mode = (
    test_error_analysis[test_error_analysis["true_class"] == 0]
    .groupby("mode_label")
    .agg(
        test_count=("mode_label", "size"),
        missed_as_allowed=("predicted_class", lambda values: int((values == 1).sum())),
    )
)
not_allowed_by_mode["recall_not_allowed"] = (
    1.0 - not_allowed_by_mode["missed_as_allowed"] / not_allowed_by_mode["test_count"]
)
not_allowed_by_mode.sort_values(["recall_not_allowed", "test_count"], ascending=[True, False])
"""),
        md("""
## Интерпретация дерева

Важность признака в дереве решений показывает, насколько признак участвовал в
уменьшении неоднородности классов. Этот показатель полезен как ориентир, но
не является доказательством физической причинности.
"""),
        md("""
## Интерпретация правил дерева решений

Правило дерева решений (decision rule) следует переводить в инженерную форму:
указать признак, порог, направление неравенства и физический смысл условия.
Например, порог по температуре должен интерпретироваться как ограничение
теплового состояния, а не как абстрактное числовое разделение.

Важность признаков (feature importance) отражает вклад признака в уменьшение
неоднородности узлов. Этот показатель зависит от обучающей выборки, выбранной
глубины дерева и набора признаков. Поэтому он является средством
интерпретации модели, но не доказательством причинной связи.
"""),
        code("""
feature_importance = (
    pd.Series(tree_model.feature_importances_, index=feature_columns)
    .sort_values(ascending=False)
)
feature_importance
"""),
        code("""
fig, ax = plt.subplots(figsize=(9, 4))
feature_importance.sort_values().plot(kind="barh", ax=ax, color="#4c78a8")
ax.set_xlabel("Относительная важность")
ax.set_title("Важность признаков в дереве решений")
plt.tight_layout()
plt.show()
"""),
        md("""
## Влияние глубины дерева

Глубина дерева определяет максимальное число последовательных условий от
корня до листа. Малое значение может привести к недообучению: модель слишком
груба и не описывает важные разделения. Слишком большая глубина может привести
к переобучению: модель запоминает частные особенности обучающей выборки.
"""),
        code("""
depth_rows = []

for depth in range(1, 8):
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    depth_rows.append(
        {
            "max_depth": depth,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_f1_allowed": f1_score(y_test, test_pred, zero_division=0),
        }
    )

depth_df = pd.DataFrame(depth_rows)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(depth_df["max_depth"], depth_df["train_accuracy"], marker="o", label="Обучающая точность")
ax.plot(depth_df["max_depth"], depth_df["test_accuracy"], marker="o", label="Тестовая точность")
ax.plot(depth_df["max_depth"], depth_df["test_f1_allowed"], marker="o", label="Тестовая F1-мера")
ax.set_xlabel("max_depth")
ax.set_ylabel("Метрика")
ax.set_ylim(0.0, 1.05)
ax.set_title("Качество классификации при разной глубине дерева")
ax.legend()
plt.tight_layout()
plt.show()

depth_df
"""),
        code("""
plt.figure(figsize=(16, 8))
plot_tree(
    tree_model,
    feature_names=feature_columns,
    class_names=["недопустимый", "допустимый"],
    filled=True,
    rounded=True,
    impurity=True,
)
plt.title("Дерево решений для классификации режимов")
plt.show()
"""),
        md("""
## Упрощенная карта решений по двум признакам

Для объяснения логики классификации полезно рассмотреть только два признака.
Такая модель хуже полной модели, но ее решение можно изобразить на плоскости.
Цвет фона показывает, какой класс дерево присваивает области признакового
пространства.
"""),
        code("""
two_feature_columns = ["temperature_c", "current_a"]
two_feature_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=8,
    random_state=RANDOM_STATE,
)
two_feature_tree.fit(X_train[two_feature_columns], y_train)

x_min, x_max = df["temperature_c"].min() - 2, df["temperature_c"].max() + 2
y_min, y_max = df["current_a"].min() - 0.5, df["current_a"].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 240),
    np.linspace(y_min, y_max, 240),
)
grid = pd.DataFrame(
    {
        "temperature_c": xx.ravel(),
        "current_a": yy.ravel(),
    }
)
zz = two_feature_tree.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 5))
ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#f3b2b0", "#b7dfb2"], alpha=0.45)
ax.scatter(df["temperature_c"], df["current_a"], c=colors, edgecolor="black", linewidth=0.2, alpha=0.75)
ax.set_xlabel("Температура, deg_C")
ax.set_ylabel("Ток, A")
ax.set_title("Карта решений упрощенного дерева по двум признакам")
plt.tight_layout()
plt.show()
"""),
        md("""
## Задание для аудиторного отчета

1. Укажите целевую переменную и признаки.
2. Приведите распределение классов.
3. Приведите значения accuracy, precision, recall и F1 отдельно для класса
   "недопустимый режим" и класса "допустимый режим".
4. Вставьте матрицу ошибок и объясните наиболее опасный тип ошибки.
5. Сформулируйте 2-3 правила дерева решений в инженерной форме.
6. Рассчитайте долю ложных разрешений:
   `dangerous_false_allowed_rate = cm[0, 1] / sum(cm[0, :])`.
7. Измените `max_depth` и объясните, как меняется качество модели.
8. Укажите, почему данное занятие является аппроксимацией инженерного правила,
   а не доказательством нового физического закона.

Ответ группы:

"""),
        code(classification_experiment_cell),
        *dataset_assignment_cells(lesson_number=3),
    ]

    if teacher:
        cells.extend(
            [
                md("""
## Методический комментарий для преподавателя

При фиксированном разбиении данных ожидаемые ориентиры для `max_depth=3`:

1. строгая модель имеет долю правильных ответов (accuracy) около 0.93;
2. полнота для класса "недопустимый режим" около 0.81, F1-мера около 0.88;
3. ожидаемая матрица ошибок строгой модели близка к `[[25, 6], [1, 73]]`;
4. наиболее значимые признаки строгого дерева обычно включают `voltage_v`,
   `temperature_c`, `speed_rpm` и `current_a`;
5. опасная ошибка - недопустимый режим, ошибочно отнесенный к допустимым.

Важно проверить, что студенты не включили в признаки `mode_label`,
`violation_count`, `current_margin_a`, `temperature_margin_c`,
`speed_margin_rpm`, признаки причин нарушения, `efficiency` или
`output_power_w`.
"""),
                code("""
print("Матрица ошибок:")
print(cm)
print("\\nВажность признаков:")
print(feature_importance.round(3))
"""),
            ]
        )

    return cells


def main() -> None:
    write_notebook(
        NOTEBOOKS_STUDENT / "01_engineering_data_student.ipynb",
        colab_bootstrap_cells() + notebook_01(teacher=False),
    )
    write_notebook(
        NOTEBOOKS_STUDENT / "02_motor_regression_student.ipynb",
        colab_bootstrap_cells() + notebook_02(teacher=False),
    )
    write_notebook(
        NOTEBOOKS_STUDENT / "03_drive_decision_tree_student.ipynb",
        colab_bootstrap_cells() + notebook_03(teacher=False),
    )
    write_notebook(
        NOTEBOOKS_TEACHER / "01_engineering_data_teacher.ipynb",
        notebook_01(teacher=True),
    )
    write_notebook(
        NOTEBOOKS_TEACHER / "02_motor_regression_teacher.ipynb",
        notebook_02(teacher=True),
    )
    write_notebook(
        NOTEBOOKS_TEACHER / "03_drive_decision_tree_teacher.ipynb",
        notebook_03(teacher=True),
    )

    print("Блокноты занятий 1-3 пересобраны.")


if __name__ == "__main__":
    main()

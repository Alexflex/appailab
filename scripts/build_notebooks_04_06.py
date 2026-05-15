"""Сборка базовых Jupyter Notebook для практических занятий 4-6."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import v4 as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_STUDENT = PROJECT_ROOT / "notebooks" / "student"
NOTEBOOKS_TEACHER = PROJECT_ROOT / "notebooks" / "teacher"


def md(source: str) -> nbformat.NotebookNode:
    return nbf.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbf.new_code_cell(dedent(source).strip())


def colab_bootstrap_cells(required_processed_files: list[str]) -> list[nbformat.NotebookNode]:
    required_repr = repr(required_processed_files)
    return [
        md(
            """
            ## Инициализация среды выполнения

            Ячейка ниже обеспечивает запуск блокнота в Google Colab и в
            локальном Jupyter Notebook. Если проект уже открыт локально,
            повторное клонирование не выполняется.
            """
        ),
        code(
            f"""
            # COLAB_BOOTSTRAP_APPailab
            from pathlib import Path
            import os
            import sys

            REQUIRED_PROCESSED_FILES = {required_repr}
            PROJECT_REPOSITORY_URL = "https://github.com/Alexflex/appailab.git"

            def find_project_root(start: Path) -> Path | None:
                for candidate in [start, *start.parents]:
                    if (candidate / "requirements-colab.txt").exists() and (candidate / "data" / "processed").exists():
                        return candidate
                return None

            project_root = find_project_root(Path.cwd())

            if project_root is None:
                try:
                    import google.colab  # type: ignore
                    IN_COLAB = True
                except Exception:
                    IN_COLAB = False

                if IN_COLAB:
                    workdir = Path("/content/appailab")
                    if not workdir.exists():
                        !git clone -q {{PROJECT_REPOSITORY_URL}} {{workdir}}
                    project_root = workdir
                    os.chdir(project_root)
                    !pip install -q -r requirements-colab.txt
                else:
                    raise FileNotFoundError(
                        "Не найден корень проекта. Откройте блокнот из репозитория appailab "
                        "или выполните git clone перед запуском."
                    )

            sys.path.insert(0, str(project_root / "src"))
            missing_files = [
                name for name in REQUIRED_PROCESSED_FILES
                if not (project_root / "data" / "processed" / name).exists()
            ]
            if missing_files:
                raise FileNotFoundError(
                    "Не найдены подготовленные CSV: " + ", ".join(missing_files)
                    + ". Выполните python scripts/generate_datasets.py."
                )

            print(f"Корень проекта: {{project_root}}")
            print("Проверенные CSV:", ", ".join(REQUIRED_PROCESSED_FILES))
            """
        ),
    ]


COMMON_IMPORTS = """
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 80)
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11

DATA_DIR = project_root / "data" / "processed"
CATALOG_04_06_FILE = DATA_DIR / "practice_04_06_dataset_catalog.csv"
ASSIGNMENTS_04_06_FILE = DATA_DIR / "practice_04_06_dataset_assignments.csv"
RANDOM_STATE = 20260507
"""


def save_notebook(cells: list[nbformat.NotebookNode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbformat.write(nb, path)


def todo_list(teacher: bool, values: list[str], name: str, comment: str) -> str:
    if teacher:
        return f"{name} = {values!r}\n"
    return (
        f"# TODO: заполните список признаков. {comment}\n"
        f"# Рекомендуемые признаки: {values!r}\n"
        f"{name} = None\n"
        f"if {name} is None:\n"
        f"    raise ValueError('Заполните {name}: {comment}')\n"
    )


def todo_value(teacher: bool, value: object, name: str, comment: str) -> str:
    if teacher:
        return f"{name} = {value!r}\n"
    return (
        f"# TODO: задайте значение параметра. {comment}\n"
        f"# Рекомендуемое значение для первого запуска: {value!r}\n"
        f"{name} = None\n"
        f"if {name} is None:\n"
        f"    raise ValueError('Заполните {name}: {comment}')\n"
    )


def todo_text(teacher: bool, value: str, name: str, comment: str) -> str:
    """Вернуть небольшую текстовую TODO-ячейку для студента."""

    if teacher:
        return f"{name} = {value!r}\nprint({name})\n"
    return (
        f"# TODO: впишите краткий текстовый ответ. {comment}\n"
        f"{name} = \"\"\n"
        f"if not {name}.strip():\n"
        f"    raise ValueError('Заполните {name}: {comment}')\n"
        f"print({name})\n"
    )


def source_section_04_06() -> nbformat.NotebookNode:
    return md(
        """
        ## Источники и проверка актуальности

        Для занятий 4-6 используются базовые локальные учебные CSV, поэтому
        выполнение блокнота не зависит от загрузки внешних архивов. Открытые
        источники ниже используются как научно-методические ориентиры для
        расширенных заданий и проверки переносимости постановки.

        1. NASA C-MAPSS Aircraft Engine Simulator Data - открытый набор
           траекторий деградации авиационных двигателей. Применение:
           временная регрессия, риск утечки между соседними точками,
           кластеризация режимов деградации. URL:
           https://data.nasa.gov/dataset/groups/c-mapss-aircraft-engine-simulator-data
        2. Mendeley Data `Partial Discharge Signals in Insulated Power
           Cables with Time-of-Arrival Annotations` - временные сигналы
           частичных разрядов с аннотациями времени прихода импульсов.
           Применение: извлечение PRPD-признаков перед классификацией. URL:
           https://data.mendeley.com/datasets/3mdgxv6zt7
        3. UCI `AI4I 2020 Predictive Maintenance Dataset` - промышленно
           мотивированный набор для предиктивного обслуживания. Применение:
           отделение сенсорных признаков от служебных кодов и признаков
           отказов перед кластеризацией. URL:
           https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
        4. NASA Prognostics Center of Excellence Data Repository - реестр
           наборов для диагностики и прогнозирования технического состояния.
           Применение: расширение задач кластеризации и анализа временных
           сценариев. URL:
           https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

        Проверка ссылок выполнена 2026-05-15. Перед использованием полного
        внешнего источника в самостоятельной работе необходимо повторно
        проверить карточку набора данных, лицензию, размер архива и формат
        файлов.
        """
    )


def dataset_assignment_cells_04_06(lesson_number: int) -> list[nbformat.NotebookNode]:
    return [
        md(
            f"""
            ## Реестр найденных наборов данных и развернутые задания

            В этом разделе используется единый реестр открытых источников для
            занятий 4-6. Реестр не загружает крупные архивы автоматически.
            Его назначение - показать, как переносить базовую учебную
            постановку на реальные или открытые исследовательские источники.

            Для занятия {lesson_number} студент должен выбрать один источник
            из таблицы ниже и описать, как на нем можно воспроизвести логику
            базового блокнота: определить признаки, целевую переменную или
            скрытую разметку, способ разбиения выборки, риск утечки данных и
            ожидаемые визуализации.
            """
        ),
        code(
            """
            dataset_catalog_04_06 = pd.read_csv(CATALOG_04_06_FILE)
            dataset_assignments_04_06 = pd.read_csv(ASSIGNMENTS_04_06_FILE)

            pd.set_option("display.max_colwidth", 120)
            display(
                dataset_catalog_04_06[
                    [
                        "dataset_id",
                        "name",
                        "object",
                        "lessons",
                        "access",
                        "risk_level",
                        "implementation_status",
                        "checked_at",
                    ]
                ]
            )
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            risk_counts = dataset_catalog_04_06["risk_level"].value_counts()
            axes[0].bar(risk_counts.index, risk_counts.values, color="#4c78a8")
            axes[0].set_title("Уровень методического риска источников")
            axes[0].set_ylabel("Число источников")

            lesson_counts = (
                dataset_assignments_04_06["lesson"]
                .astype(str)
                .value_counts()
                .sort_index()
            )
            axes[1].bar(lesson_counts.index, lesson_counts.values, color="#f58518")
            axes[1].set_title("Число развернутых заданий по занятиям 4-6")
            axes[1].set_xlabel("Номер занятия")
            axes[1].set_ylabel("Число заданий")

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            f"""
            lesson_assignments = dataset_assignments_04_06[
                dataset_assignments_04_06["lesson"].astype(str) == "{lesson_number}"
            ].copy()

            display(
                lesson_assignments[
                    [
                        "assignment_id",
                        "assignment_title",
                        "implementation_status",
                        "dataset_structure",
                        "theory_block",
                        "practice_block",
                        "recommended_visualizations",
                        "expected_artifacts",
                        "control_questions",
                        "risk_note",
                    ]
                ]
            )
            """
        ),
        md(
            """
            ## Индивидуальное расширенное задание

            Выберите один источник из таблицы выше и заполните в отчете
            отдельный подраздел:

            1. объект исследования и единица наблюдения;
            2. какие столбцы являются измеряемыми признаками;
            3. какая величина является целевой переменной или скрытой
               диагностической разметкой;
            4. какие столбцы нельзя использовать как признаки из-за риска
               утечки данных;
            5. какой способ разбиения выборки является корректным;
            6. какие 2-3 графика нужно построить в первую очередь.

            Если полный архив не скачивался, это нужно явно указать. В таком
            случае результатом считается методически корректная постановка
            расширенного задания, а не численное обучение модели.
            """
        ),
    ]


def build_practice_04(teacher: bool) -> list[nbformat.NotebookNode]:
    features = [
        "altitude_m",
        "air_density_kg_m3",
        "ambient_temp_c",
        "cooling_air_speed_mps",
        "speed_rpm",
        "torque_nm",
        "voltage_v",
        "current_a",
    ]
    cells = [
        md(
            """
            # Практическое занятие 4. Прогноз температуры электропривода высотной платформы

            Цель занятия - построить и сравнить простую физически
            интерпретируемую модель, линейную регуляризованную модель,
            случайный лес и градиентный бустинг для прогноза температуры
            обмотки электродвигателя.

            HAPS (High Altitude Platform Station) - высотная
            псевдоспутниковая платформа. В инженерном смысле это летательный
            аппарат длительного пребывания в стратосфере. Для такого объекта
            тепловой режим электропривода важен из-за ограниченных запасов
            массы, энергии и охлаждения.

            **Задача студента.** Не требуется писать сложный код с нуля.
            Необходимо последовательно выполнить ячейки, заполнить несколько
            явно отмеченных учебных параметров, сравнить модели и объяснить,
            какие признаки связаны с нагревом, а какие столбцы являются
            диагностическими и не должны попадать в базовую модель.
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_04_haps_thermal_features.csv",
                "practice_04_haps_thermal_diagnostics.csv",
                "practice_04_06_dataset_catalog.csv",
                "practice_04_06_dataset_assignments.csv",
            ]
        ),
        code(COMMON_IMPORTS),
        source_section_04_06(),
        md(
            """
            ## Теоретический блок

            Тепловая модель первого порядка описывает инерционное изменение
            температуры:

            $$
            T_{k+1} = T_k + \\frac{\\Delta t}{\\tau}
            \\left(T_{amb,k} + R_{th} P_{loss,k} - T_k\\right),
            $$

            где `T` - температура обмотки, `T_amb` - температура окружающей
            среды, `R_th` - тепловое сопротивление, `tau` - тепловая
            постоянная времени, `P_loss` - суммарные потери.

            Случайный лес (Random Forest) - ансамбль деревьев решений,
            усредняющий прогнозы многих деревьев. Градиентный бустинг
            (Gradient Boosting) - последовательный ансамбль, в котором каждая
            следующая модель уточняет ошибки предыдущих. Оба метода полезны
            для нелинейных зависимостей, но требуют контроля переобучения и
            осторожной интерпретации важности признаков.
            """
        ),
        md(
            """
            ## Последовательность работы

            В этой работе используется воспроизводимый pipeline
            (последовательность обработки данных):

            1. загрузить feature-CSV и diagnostics-CSV;
            2. проверить физический смысл столбцов;
            3. построить разведочные графики;
            4. выбрать безопасные признаки;
            5. обучить физическую базовую модель и модели машинного обучения;
            6. сравнить метрики и остатки;
            7. разобрать антипример утечки данных;
            8. сформулировать инженерный вывод.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(12, 3.8))
            ax.axis("off")
            boxes = [
                ("Электрическая\\nмощность UI", 0.05),
                ("Полезная\\nмеханическая\\nмощность Mω", 0.27),
                ("Потери\\nP_loss", 0.49),
                ("Нагрев\\nобмотки", 0.68),
                ("Охлаждение\\nвоздушным\\nпотоком", 0.86),
            ]
            for text, x in boxes:
                ax.text(
                    x,
                    0.55,
                    text,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#f4f7fb", edgecolor="#4c78a8"),
                    transform=ax.transAxes,
                )
            for x0, x1 in [(0.14, 0.22), (0.36, 0.44), (0.58, 0.64), (0.77, 0.82)]:
                ax.annotate(
                    "",
                    xy=(x1, 0.55),
                    xytext=(x0, 0.55),
                    xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
                )
            ax.text(
                0.49,
                0.18,
                "Учебный прокси-признак: P_loss_proxy = U I - Mω",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Схема теплового баланса электропривода")
            plt.show()
            """
        ),
        code(
            """
            FEATURES_FILE = DATA_DIR / "practice_04_haps_thermal_features.csv"
            DIAGNOSTICS_FILE = DATA_DIR / "practice_04_haps_thermal_diagnostics.csv"

            df = pd.read_csv(FEATURES_FILE)
            diagnostics_df = pd.read_csv(DIAGNOSTICS_FILE)
            full_df = df.merge(diagnostics_df, on="sample_id", validate="one_to_one")

            display(df.head())
            print("Размер feature-таблицы:", df.shape)
            print("Размер diagnostics-таблицы:", diagnostics_df.shape)
            """
        ),
        md(
            """
            ## Паспорт набора данных

            `feature-CSV` содержит только признаки, допустимые для базового
            моделирования, и целевую переменную `winding_temp_c`.
            `diagnostics-CSV` содержит расчетные тепловые величины:
            суммарные потери, тепловое сопротивление, постоянную времени,
            стационарную температуру и запас до теплового предела.

            В базовую модель запрещено автоматически включать
            `steady_state_temp_c`, `temperature_margin_c` и `is_overheated`,
            так как эти столбцы прямо раскрывают способ формирования тепловой
            цели или ее ограничений.
            """
        ),
        md(
            """
            Важное правило занятия: diagnostics-CSV разрешается использовать
            для объяснения физики и проверки выводов, но не как автоматический
            источник входных признаков. Такое разделение защищает модель от
            утечки данных (data leakage), то есть попадания в признаки
            информации, которая недоступна в реальной задаче прогноза.
            """
        ),
        code(
            """
            display(df.describe().T)
            print("Пропуски по столбцам:")
            display(df.isna().sum().to_frame("missing_count"))
            print("Диапазон температуры обмотки:")
            display(df["winding_temp_c"].describe())
            """
        ),
        md("## Разведочный анализ данных"),
        code(
            """
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            axes = axes.ravel()

            axes[0].hist(df["winding_temp_c"], bins=28, color="#4c78a8", edgecolor="white")
            axes[0].set_title("Распределение температуры обмотки")
            axes[0].set_xlabel("Температура, deg_C")

            axes[1].scatter(df["current_a"], df["winding_temp_c"], alpha=0.65, s=18)
            axes[1].set_title("Температура и ток")
            axes[1].set_xlabel("Ток, A")
            axes[1].set_ylabel("Температура, deg_C")

            axes[2].scatter(df["cooling_air_speed_mps"], df["winding_temp_c"], alpha=0.65, s=18)
            axes[2].set_title("Температура и скорость охлаждающего потока")
            axes[2].set_xlabel("Скорость потока, m/s")
            axes[2].set_ylabel("Температура, deg_C")

            profile_means = df.groupby("profile_id")["winding_temp_c"].mean()
            axes[3].bar(profile_means.index.astype(str), profile_means.values, color="#72b7b2")
            axes[3].set_title("Средняя температура по профилям")
            axes[3].set_xlabel("profile_id")
            axes[3].set_ylabel("Температура, deg_C")

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            На следующем графике показана динамика температуры по профилям.
            Это не полноценный временной ряд полета, но упорядоченность
            `time_s` внутри каждого `profile_id` позволяет увидеть тепловую
            инерцию: температура меняется плавнее, чем электрическая нагрузка.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(11, 5))
            for profile_id, group in df.groupby("profile_id"):
                ax.plot(group["time_s"], group["winding_temp_c"], label=f"profile {profile_id}", linewidth=1.8)
            ax.set_title("Профили температуры обмотки во времени")
            ax.set_xlabel("Время внутри профиля, s")
            ax.set_ylabel("Температура обмотки, deg_C")
            ax.legend(ncol=3)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            График ниже использует `total_loss_w` из diagnostics-CSV только
            для интерпретации. В реальном прогнозе эта величина может быть
            неизвестна заранее или рассчитана из скрытых параметров объекта.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(
                full_df["total_loss_w"],
                full_df["winding_temp_c"],
                c=full_df["cooling_air_speed_mps"],
                cmap="viridis",
                s=24,
                alpha=0.75,
            )
            ax.set_title("Температура, потери и охлаждающий поток")
            ax.set_xlabel("Суммарные потери из diagnostics, W")
            ax.set_ylabel("Температура обмотки, deg_C")
            fig.colorbar(scatter, ax=ax, label="Скорость охлаждающего потока, m/s")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_columns].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im, ax=ax, label="Коэффициент корреляции")
            ax.set_title("Корреляционная матрица признаков занятия 4")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Выбор признаков и разбиение выборки

            Для начального занятия используется случайное разбиение с
            сохранением доли каждого `profile_id` в обучающей и тестовой
            выборках. Это упрощает сравнение моделей. В задачах строгого
            прогнозирования временных рядов следует дополнительно проверять
            групповое или хронологическое разбиение, чтобы соседние точки
            одного испытания не создавали утечку информации.
            """
        ),
        code(
            todo_list(
                teacher,
                features,
                "thermal_features",
                "используйте только измеряемые признаки из feature-CSV",
            )
            + dedent(
                """
            target_column = "winding_temp_c"
            forbidden_columns = {"winding_temp_c", "steady_state_temp_c", "temperature_margin_c", "is_overheated"}
            leaked = forbidden_columns.intersection(thermal_features)
            if leaked:
                raise ValueError(f"Обнаружена утечка данных: {sorted(leaked)}")

            train_idx, test_idx = train_test_split(
                df.index,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=df["profile_id"],
            )

            X_train = df.loc[train_idx, thermal_features]
            X_test = df.loc[test_idx, thermal_features]
            y_train = df.loc[train_idx, target_column]
            y_test = df.loc[test_idx, target_column]

            print("Train:", X_train.shape, "Test:", X_test.shape)
            display(df.loc[test_idx, "profile_id"].value_counts().sort_index().to_frame("test_count"))
            """
            )
        ),
        md(
            """
            ## Проверка риска утечки между близкими режимами

            Случайное разбиение удобно для первого сравнения моделей, но оно
            может завышать качество, если соседние точки одного профиля
            одновременно попадают в обучение и тест. Поэтому ниже
            дополнительно показано групповое разбиение: часть `profile_id`
            полностью оставляется для теста.
            """
        ),
        code(
            """
            group_test_profiles = [5, 6]
            group_train_mask = ~df["profile_id"].isin(group_test_profiles)
            group_test_mask = df["profile_id"].isin(group_test_profiles)

            split_diagnostics = []
            for split_name, train_index, test_index in [
                ("random_stratified", train_idx, test_idx),
                ("group_holdout", df.index[group_train_mask], df.index[group_test_mask]),
            ]:
                split_model = RandomForestRegressor(
                    n_estimators=160,
                    max_depth=8,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                )
                split_model.fit(df.loc[train_index, thermal_features], df.loc[train_index, target_column])
                split_pred = split_model.predict(df.loc[test_index, thermal_features])
                split_diagnostics.append(
                    {
                        "split": split_name,
                        "test_size": len(test_index),
                        "RMSE_deg_C": np.sqrt(mean_squared_error(df.loc[test_index, target_column], split_pred)),
                        "R2": r2_score(df.loc[test_index, target_column], split_pred),
                    }
                )
            split_diagnostics_df = pd.DataFrame(split_diagnostics)
            display(split_diagnostics_df)
            """
        ),
        md(
            """
            ## Простая физически мотивированная модель

            Физически мотивированная модель не является строгой тепловой
            симуляцией. Она использует приближенный признак потерь, рассчитанный
            из доступных измерений:

            $$
            P_{loss,proxy} = U I - M \\omega.
            $$

            Далее линейная регрессия оценивает связь температуры с
            температурой окружающей среды, прокси-потерями и охлаждением.
            """
        ),
        code(
            """
            def make_physics_features(data: pd.DataFrame) -> pd.DataFrame:
                omega = 2.0 * np.pi * data["speed_rpm"] / 60.0
                loss_proxy = data["voltage_v"] * data["current_a"] - data["torque_nm"] * omega
                return pd.DataFrame(
                    {
                        "ambient_temp_c": data["ambient_temp_c"],
                        "loss_proxy_w": loss_proxy,
                        "cooling_inverse": 1.0 / np.sqrt(data["cooling_air_speed_mps"].clip(lower=1.0)),
                    },
                    index=data.index,
                )

            physics_model = LinearRegression()
            physics_model.fit(make_physics_features(df.loc[train_idx]), y_train)
            physics_pred = physics_model.predict(make_physics_features(df.loc[test_idx]))
            """
        ),
        code(
            todo_text(
                teacher,
                "Прокси-потери P_loss_proxy показывают часть входной электрической мощности, которая не превратилась в полезную механическую мощность и поэтому связана с нагревом.",
                "physics_proxy_explanation",
                "объясните физический смысл P_loss_proxy = U I - M omega",
            )
        ),
        md("## Обучение моделей машинного обучения"),
        code(
            todo_value(teacher, 8, "forest_max_depth", "рекомендуемый диапазон max_depth случайного леса: 3..10")
            + dedent(
            """
            models = {
                "physics_proxy_linear": physics_model,
                "ridge": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=1.0)),
                    ]
                ),
                "random_forest": RandomForestRegressor(
                    n_estimators=250,
                    max_depth=forest_max_depth,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=180,
                    learning_rate=0.045,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            }

            predictions = {"physics_proxy_linear": physics_pred}
            for name, model in models.items():
                if name == "physics_proxy_linear":
                    continue
                model.fit(X_train, y_train)
                predictions[name] = model.predict(X_test)

            metrics_rows = []
            for name, pred in predictions.items():
                metrics_rows.append(
                    {
                        "model": name,
                        "MAE_deg_C": mean_absolute_error(y_test, pred),
                        "RMSE_deg_C": np.sqrt(mean_squared_error(y_test, pred)),
                        "R2": r2_score(y_test, pred),
                    }
                )
            metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE_deg_C")
            display(metrics_df)
            """
            )
        ),
        code(
            """
            best_model_name = metrics_df.iloc[0]["model"]
            best_pred = predictions[best_model_name]

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].scatter(y_test, best_pred, alpha=0.70, s=20)
            lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
            axes[0].plot(lims, lims, color="black", linestyle="--")
            axes[0].set_xlabel("Измеренная температура, deg_C")
            axes[0].set_ylabel("Прогноз, deg_C")
            axes[0].set_title(f"Лучшая модель: {best_model_name}")

            residuals = y_test - best_pred
            axes[1].scatter(df.loc[test_idx, "speed_rpm"], residuals, alpha=0.70, s=20)
            axes[1].axhline(0.0, color="black", linestyle="--")
            axes[1].set_xlabel("Частота вращения, rpm")
            axes[1].set_ylabel("Остаток, deg_C")
            axes[1].set_title("Остатки по частоте вращения")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            todo_text(
                teacher,
                "По RMSE рационально выбрать модель с минимальной ошибкой на тестовой выборке; дополнительно следует проверить остатки, чтобы убедиться в отсутствии выраженного смещения по режимным признакам.",
                "best_model_interpretation",
                "укажите лучшую модель по RMSE и объясните, почему одной метрики недостаточно",
            )
        ),
        md(
            """
            ## Анализ остатков

            Остаток - разность между измеренной температурой и прогнозом.
            Если остатки систематически зависят от высоты, тока или скорости
            охлаждения, значит модель не полностью описывает соответствующий
            физический фактор.
            """
        ),
        code(
            """
            residual_analysis_df = df.loc[test_idx, ["altitude_m", "current_a", "cooling_air_speed_mps"]].copy()
            residual_analysis_df["residual_deg_C"] = y_test - best_pred

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, column, title in zip(
                axes,
                ["altitude_m", "current_a", "cooling_air_speed_mps"],
                ["Остаток по высоте", "Остаток по току", "Остаток по охлаждению"],
            ):
                ax.scatter(residual_analysis_df[column], residual_analysis_df["residual_deg_C"], s=20, alpha=0.70)
                ax.axhline(0.0, color="black", linestyle="--")
                ax.set_title(title)
                ax.set_xlabel(column)
                ax.set_ylabel("Остаток, deg_C")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            todo_text(
                teacher,
                "Если остатки примерно симметричны вокруг нуля и не образуют наклонной полосы, явного систематического смещения по выбранному признаку не видно.",
                "residual_plot_interpretation",
                "выберите один график остатков и кратко опишите, есть ли на нем систематическое смещение",
            )
        ),
        code(
            """
            rf_model = models["random_forest"]
            importance_df = pd.DataFrame(
                {
                    "feature": thermal_features,
                    "importance": rf_model.feature_importances_,
                }
            ).sort_values("importance", ascending=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(importance_df["feature"], importance_df["importance"], color="#59a14f")
            ax.set_title("Важность признаков случайного леса")
            ax.set_xlabel("Относительная важность")
            plt.tight_layout()
            plt.show()

            display(importance_df.sort_values("importance", ascending=False))
            """
        ),
        md(
            """
            ## Перестановочная важность признаков

            Перестановочная важность (permutation importance) показывает,
            насколько ухудшается качество модели, если значения одного
            признака случайно перемешать в тестовой выборке. Такой прием
            удобен для объяснения: если после перемешивания признака ошибка
            резко растет, значит модель сильно опиралась на этот признак.
            """
        ),
        code(
            """
            permutation_result = permutation_importance(
                rf_model,
                X_test,
                y_test,
                n_repeats=8,
                random_state=RANDOM_STATE,
                scoring="neg_root_mean_squared_error",
            )
            permutation_importance_df = pd.DataFrame(
                {
                    "feature": thermal_features,
                    "importance_mean": permutation_result.importances_mean,
                    "importance_std": permutation_result.importances_std,
                }
            ).sort_values("importance_mean", ascending=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(
                permutation_importance_df["feature"],
                permutation_importance_df["importance_mean"],
                xerr=permutation_importance_df["importance_std"],
                color="#9c755f",
            )
            ax.set_title("Перестановочная важность признаков")
            ax.set_xlabel("Рост RMSE при перемешивании признака, deg_C")
            plt.tight_layout()
            plt.show()
            display(permutation_importance_df.sort_values("importance_mean", ascending=False))
            """
        ),
        md(
            """
            ## Демонстрация утечки данных

            > **Внимание. АНТИПРИМЕР - НЕ ИСПОЛЬЗОВАТЬ КАК РАБОЧУЮ МОДЕЛЬ.**
            > В следующей ячейке в признаки намеренно добавляется
            > `steady_state_temp_c`. Этот столбец рассчитан из скрытой
            > физической модели генератора и поэтому является диагностической
            > подсказкой, а не обычным измеряемым признаком.
            """
        ),
        code(
            """
            leakage_df = df.merge(
                diagnostics_df[["sample_id", "steady_state_temp_c", "temperature_margin_c"]],
                on="sample_id",
                validate="one_to_one",
            )
            leakage_features = thermal_features + ["steady_state_temp_c"]

            X_train_leak = leakage_df.loc[train_idx, leakage_features]
            X_test_leak = leakage_df.loc[test_idx, leakage_features]
            leakage_model = RandomForestRegressor(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
            )
            leakage_model.fit(X_train_leak, y_train)
            leakage_pred = leakage_model.predict(X_test_leak)
            print("R2 строгой лучшей модели:", round(float(metrics_df.iloc[0]["R2"]), 4))
            print("R2 модели с утечкой:", round(r2_score(y_test, leakage_pred), 4))
            """
        ),
        code(
            todo_text(
                teacher,
                "steady_state_temp_c является утечкой, потому что это расчетная величина скрытой тепловой модели, близкая к целевой температуре и недоступная как независимый датчик в момент прогноза.",
                "thermal_leakage_explanation",
                "объясните, почему steady_state_temp_c нельзя использовать как базовый признак",
            )
        ),
        md("## Самостоятельный эксперимент"),
        code(
            todo_value(teacher, 6, "experiment_max_depth", "рекомендуемый диапазон max_depth: 3..10")
            + dedent(
                """
            experiment_model = RandomForestRegressor(
                n_estimators=160,
                max_depth=experiment_max_depth,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
            )
            experiment_model.fit(X_train, y_train)
            experiment_pred = experiment_model.predict(X_test)
            print("MAE:", round(mean_absolute_error(y_test, experiment_pred), 3))
            print("RMSE:", round(np.sqrt(mean_squared_error(y_test, experiment_pred)), 3))
            print("R2:", round(r2_score(y_test, experiment_pred), 4))
            """
            )
        ),
        *dataset_assignment_cells_04_06(4),
        md(
            """
            ## Мини-задание по открытому источнику

            NASA C-MAPSS Aircraft Engine Simulator Data - открытый набор
            данных по моделированию деградации авиационного двигателя. Для
            самостоятельного расширения не требуется скачивать архив прямо
            сейчас. В отчете укажите:

            1. какие столбцы могли бы быть признаками;
            2. какую величину можно считать целевой переменной;
            3. почему для такого источника опасно случайно перемешивать
               соседние временные точки;
            4. какую метрику качества следует использовать для прогноза
               температуры или остаточного ресурса.
            """
        ),
        md(
            """
            ## Задание для отчета

            1. Объясните, почему случайное разбиение нужно сравнивать с
               групповым разбиением по профилям.
            2. Сравните физически мотивированную модель, Ridge-регрессию,
               случайный лес и градиентный бустинг.
            3. Укажите две наиболее важные величины для прогноза температуры.
            4. Объясните, почему важность признака в случайном лесе не равна
               строгой физической причинности.
            5. Опишите антипример утечки данных и не используйте его метрики
               как основной результат.
            6. Выполните мини-задание по NASA C-MAPSS на уровне постановки
               задачи без загрузки архива.

            Открытые источники для расширения: NASA C-MAPSS Aircraft Engine
            Simulator Data и открытые наборы температурного моделирования
            электромеханических систем. Их следует применять только после
            отдельной проверки структуры временных рядов и правил разбиения.
            """
        ),
    ]
    return cells


def build_practice_05(teacher: bool) -> list[nbformat.NotebookNode]:
    features = [
        "voltage_kv",
        "frequency_hz",
        "phase_mean_deg",
        "phase_std_deg",
        "pulse_count",
        "mean_charge_pc",
        "max_charge_pc",
        "charge_iqr_pc",
        "repetition_rate_hz",
        "positive_negative_ratio",
        "waveform_rise_ns",
        "waveform_width_ns",
        "prpd_entropy",
    ]
    cells = [
        md(
            """
            # Практическое занятие 5. Классификация частичных разрядов по PRPD-признакам

            Цель занятия - научиться строить диагностическую классификацию
            состояния изоляции по признакам частичных разрядов и сравнить
            несколько базовых классификаторов.

            Частичный разряд - локальный электрический разряд, который
            возникает в части изоляционной системы и не полностью перекрывает
            промежуток между электродами. PRPD (Phase Resolved Partial
            Discharge) - фазово-разрешенное представление частичных разрядов,
            связывающее импульсы с фазой питающего напряжения.

            **Задача студента.** Нужно выбрать безопасные PRPD-признаки,
            обучить несколько классификаторов, сравнить метрики и объяснить,
            какие диагностические ошибки наиболее существенны. Основной
            результат - корректная интерпретация графиков, матрицы ошибок и
            антипримеров утечки данных.
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_05_partial_discharge_features.csv",
                "practice_05_partial_discharge_diagnostics.csv",
                "practice_04_06_dataset_catalog.csv",
                "practice_04_06_dataset_assignments.csv",
            ]
        ),
        code(COMMON_IMPORTS),
        source_section_04_06(),
        md(
            """
            ## Теоретический блок

            Метод опорных векторов (Support Vector Machine, SVM) строит
            разделяющую поверхность между классами. Для линейного SVM эта
            поверхность является гиперплоскостью. Для SVM с радиальной
            базисной функцией (Radial Basis Function, RBF) граница может быть
            нелинейной.

            Логистическая регрессия (Logistic Regression) служит простой
            базовой моделью классификации. Случайный лес и дерево решений
            позволяют получить более гибкую нелинейную границу, но требуют
            контроля переобучения.

            Для многоклассовой диагностики важны не только accuracy, но и
            macro-precision, macro-recall и macro-F1. Приставка macro означает,
            что метрика сначала считается по каждому классу, а затем
            усредняется без учета размера класса.
            """
        ),
        md(
            """
            ## Последовательность работы

            Pipeline занятия:

            1. загрузить таблицу PRPD-признаков;
            2. проверить классы и распределения признаков;
            3. выбрать безопасные признаки без diagnostic-столбцов;
            4. выполнить масштабирование признаков;
            5. обучить несколько классификаторов;
            6. сравнить accuracy и macro-F1;
            7. разобрать матрицу ошибок;
            8. проверить антипримеры: отсутствие масштабирования и утечку
               `risk_score`.
            """
        ),
        code(
            """
            FEATURES_FILE = DATA_DIR / "practice_05_partial_discharge_features.csv"
            DIAGNOSTICS_FILE = DATA_DIR / "practice_05_partial_discharge_diagnostics.csv"

            df = pd.read_csv(FEATURES_FILE)
            diagnostics_df = pd.read_csv(DIAGNOSTICS_FILE)
            full_df = df.merge(diagnostics_df, on="sample_id", validate="one_to_one")

            display(df.head())
            print("Размер feature-таблицы:", df.shape)
            print("Классы:", sorted(df["defect_class"].unique()))
            """
        ),
        md(
            """
            ## Схематическая PRPD-диаграмма

            PRPD-диаграмма показывает, в каких фазовых областях появляются
            импульсы и какова их амплитуда. В учебном наборе используются уже
            извлеченные признаки, но диаграмма ниже помогает связать
            табличные признаки с исходной физической картиной.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 5))
            colors_by_class = {
                "no_pd": "#4c78a8",
                "corona": "#f28e2b",
                "surface": "#59a14f",
                "internal": "#e15759",
            }
            for label, group in df.groupby("defect_class"):
                ax.scatter(
                    group["phase_mean_deg"],
                    group["max_charge_pc"],
                    s=22,
                    alpha=0.65,
                    label=label,
                    color=colors_by_class[label],
                )
            ax.set_xlim(0, 360)
            ax.set_xlabel("Фаза питающего напряжения, deg")
            ax.set_ylabel("Максимальный кажущийся заряд, pC")
            ax.set_title("Учебная PRPD-диаграмма: фаза и амплитуда разрядов")
            ax.legend(title="Класс")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Структура данных

            В feature-CSV находится целевая переменная `defect_class`.
            Остальные столбцы описывают статистические признаки PRPD:
            фазовое положение, разброс фазы, число импульсов, кажущийся заряд,
            повторяемость и параметры формы импульса.

            В diagnostics-CSV находятся производные диагностические величины:
            числовой код класса, бинарный признак наличия разряда и риск-оценка.
            Эти столбцы не должны использоваться как входные признаки базовой
            модели.
            """
        ),
        md(
            """
            Первый график ниже показывает баланс классов, второй - разделение
            классов по двум физически интерпретируемым признакам: фазе и
            среднему кажущемуся заряду. Если классы хорошо разделены уже на
            таком графике, высокая точность модели ожидаема и не должна
            восприниматься как универсальная гарантия для реальных данных.
            """
        ),
        code(
            """
            display(df.describe(include="all").T)
            class_counts = df["defect_class"].value_counts().sort_index()
            display(class_counts.to_frame("count"))

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].bar(class_counts.index, class_counts.values, color="#4c78a8")
            axes[0].set_title("Распределение классов")
            axes[0].set_xlabel("Класс")
            axes[0].set_ylabel("Число наблюдений")

            for label, group in df.groupby("defect_class"):
                axes[1].scatter(group["phase_mean_deg"], group["mean_charge_pc"], s=18, alpha=0.65, label=label)
            axes[1].set_title("PRPD-пространство: фаза и средний заряд")
            axes[1].set_xlabel("Средняя фаза, deg")
            axes[1].set_ylabel("Средний кажущийся заряд, pC")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            corr_features = [
                "phase_std_deg",
                "pulse_count",
                "mean_charge_pc",
                "max_charge_pc",
                "repetition_rate_hz",
                "prpd_entropy",
            ]
            pd.plotting.scatter_matrix(df[corr_features], figsize=(12, 12), diagonal="hist", alpha=0.35)
            plt.suptitle("Парные зависимости основных PRPD-признаков", y=1.02)
            plt.show()
            """
        ),
        md("## Выбор признаков и подготовка выборок"),
        code(
            todo_list(
                teacher,
                features,
                "pd_features",
                "не включайте defect_code, risk_score, high_risk и has_partial_discharge",
            )
            + dedent(
                """
            target_column = "defect_class"
            forbidden_columns = {"defect_class", "defect_code", "risk_score", "high_risk", "has_partial_discharge"}
            leaked = forbidden_columns.intersection(pd_features)
            if leaked:
                raise ValueError(f"Обнаружена утечка данных: {sorted(leaked)}")

            X = df[pd_features]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            print("Train:", X_train.shape, "Test:", X_test.shape)
            display(y_test.value_counts().sort_index().to_frame("test_count"))
            """
            )
        ),
        code(
            todo_text(
                teacher,
                "StandardScaler нужен, потому что SVM использует расстояния между объектами, а признаки измеряются в разных единицах: градусах, пикокулонах, наносекундах и герцах.",
                "scaler_explanation",
                "объясните, зачем SVM требуется масштабирование признаков",
            )
        ),
        md("## Обучение классификаторов"),
        code(
            """
            classifiers = {
                "logistic_regression": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                    ]
                ),
                "decision_tree": DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                "random_forest": RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                "svm_linear": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", SVC(kernel="linear", C=1.0, class_weight="balanced")),
                    ]
                ),
                "svm_rbf": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced")),
                    ]
                ),
            }

            predictions = {}
            metric_rows = []
            for name, model in classifiers.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                predictions[name] = pred
                metric_rows.append(
                    {
                        "model": name,
                        "accuracy": accuracy_score(y_test, pred),
                        "precision_macro": precision_score(y_test, pred, average="macro"),
                        "recall_macro": recall_score(y_test, pred, average="macro"),
                        "f1_macro": f1_score(y_test, pred, average="macro"),
                    }
                )

            metrics_df = pd.DataFrame(metric_rows).sort_values("f1_macro", ascending=False)
            display(metrics_df)
            """
        ),
        code(
            todo_text(
                teacher,
                "Для этой диагностической задачи рационально выбирать модель по macro-F1, потому что эта метрика учитывает качество по каждому классу, а не только общую долю правильных ответов.",
                "best_classifier_interpretation",
                "выберите модель по macro-F1 и кратко обоснуйте выбор",
            )
        ),
        md(
            """
            ## Сравнение balanced и imbalanced выборок

            Accuracy может выглядеть высокой на несбалансированной выборке,
            даже если редкий дефект распознается хуже. Поэтому дополнительно
            сравним модель на исходной сбалансированной обучающей выборке и
            на искусственно несбалансированной подвыборке.
            """
        ),
        code(
            """
            train_df_for_imbalance = df.loc[X_train.index].copy()
            imbalanced_parts = []
            for label, group in train_df_for_imbalance.groupby("defect_class"):
                keep_count = len(group) if label == "no_pd" else max(18, len(group) // 3)
                imbalanced_parts.append(group.sample(n=keep_count, random_state=RANDOM_STATE))
            imbalanced_train_df = pd.concat(imbalanced_parts, ignore_index=False)

            balanced_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000)),
                ]
            )
            imbalanced_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000)),
                ]
            )
            balanced_model.fit(X_train, y_train)
            imbalanced_model.fit(imbalanced_train_df[pd_features], imbalanced_train_df[target_column])
            balanced_pred = balanced_model.predict(X_test)
            imbalanced_pred = imbalanced_model.predict(X_test)
            imbalance_metrics = pd.DataFrame(
                [
                    {
                        "training_set": "balanced",
                        "accuracy": accuracy_score(y_test, balanced_pred),
                        "macro_F1": f1_score(y_test, balanced_pred, average="macro"),
                    },
                    {
                        "training_set": "artificially_imbalanced",
                        "accuracy": accuracy_score(y_test, imbalanced_pred),
                        "macro_F1": f1_score(y_test, imbalanced_pred, average="macro"),
                    },
                ]
            )
            display(imbalance_metrics)
            """
        ),
        code(
            """
            best_model_name = metrics_df.iloc[0]["model"]
            best_pred = predictions[best_model_name]
            labels = sorted(y.unique())

            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                best_pred,
                labels=labels,
                xticks_rotation=45,
                cmap="Blues",
                ax=ax,
            )
            ax.set_title(f"Матрица ошибок: {best_model_name}")
            plt.tight_layout()
            plt.show()

            print(classification_report(y_test, best_pred, digits=3))
            """
        ),
        md(
            """
            ## Разбор диагностических ошибок

            Матрица ошибок должна интерпретироваться не только формально, но
            и инженерно. Ошибка между `corona` и `surface` имеет один смысл,
            а пропуск класса `internal` может быть существенно опаснее,
            поскольку внутренние дефекты часто связаны с объемом изоляции.
            """
        ),
        code(
            """
            error_df = pd.DataFrame({"true": y_test, "predicted": best_pred})
            error_df["is_error"] = error_df["true"] != error_df["predicted"]
            error_summary = (
                error_df.groupby("true")
                .agg(test_count=("true", "size"), error_count=("is_error", "sum"))
                .assign(error_rate=lambda data: data["error_count"] / data["test_count"])
                .sort_values("error_rate", ascending=False)
            )
            display(error_summary)
            """
        ),
        code(
            todo_text(
                teacher,
                "В эталонном запуске классы почти не путаются; если ошибка появилась, необходимо указать истинный класс, предсказанный класс и возможную физическую причину близости PRPD-признаков.",
                "diagnostic_error_interpretation",
                "опишите один тип диагностической ошибки по матрице ошибок",
            )
        ),
        md(
            """
            ## Учебная граница SVM по двум признакам

            Следующая схема не заменяет многомерную модель. Она нужна, чтобы
            визуально показать идею SVM: алгоритм строит границу между
            областями классов в пространстве признаков. Здесь используются
            только `phase_mean_deg` и `mean_charge_pc`.
            """
        ),
        code(
            """
            svm_2d_features = ["phase_mean_deg", "mean_charge_pc"]
            svm_2d = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", C=3.0, gamma="scale")),
                ]
            )
            svm_2d.fit(X_train[svm_2d_features], y_train)
            x_min, x_max = df["phase_mean_deg"].min() - 10, df["phase_mean_deg"].max() + 10
            y_min, y_max = df["mean_charge_pc"].min() - 10, df["mean_charge_pc"].max() + 10
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
            grid = pd.DataFrame(
                {
                    "phase_mean_deg": xx.ravel(),
                    "mean_charge_pc": yy.ravel(),
                }
            )
            zz = svm_2d.predict(grid).reshape(xx.shape)
            label_to_int = {label: index for index, label in enumerate(sorted(y.unique()))}
            zz_int = np.vectorize(label_to_int.get)(zz)

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.contourf(xx, yy, zz_int, levels=len(label_to_int), alpha=0.20, cmap="tab10")
            for label, group in df.groupby("defect_class"):
                ax.scatter(group["phase_mean_deg"], group["mean_charge_pc"], s=18, alpha=0.65, label=label)
            ax.set_xlabel("Средняя фаза, deg")
            ax.set_ylabel("Средний кажущийся заряд, pC")
            ax.set_title("Учебная граница SVM по двум PRPD-признакам")
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            rf_model = classifiers["random_forest"]
            importance_df = pd.DataFrame(
                {
                    "feature": pd_features,
                    "importance": rf_model.feature_importances_,
                }
            ).sort_values("importance", ascending=True)

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(importance_df["feature"], importance_df["importance"], color="#f28e2b")
            ax.set_title("Важность признаков случайного леса")
            ax.set_xlabel("Относительная важность")
            plt.tight_layout()
            plt.show()
            display(importance_df.sort_values("importance", ascending=False))
            """
        ),
        md(
            """
            ## Антипример: SVM без масштабирования

            SVM чувствителен к расстояниям. Если признаки с разными единицами
            измерения не масштабировать, признаки с большими численными
            диапазонами получают непропорционально высокий вклад.
            """
        ),
        code(
            """
            svm_without_scaling = SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced")
            svm_without_scaling.fit(X_train, y_train)
            no_scaling_pred = svm_without_scaling.predict(X_test)
            scaling_comparison = pd.DataFrame(
                [
                    {
                        "model": "svm_rbf_with_scaler",
                        "accuracy": accuracy_score(y_test, predictions["svm_rbf"]),
                        "macro_F1": f1_score(y_test, predictions["svm_rbf"], average="macro"),
                    },
                    {
                        "model": "svm_rbf_without_scaler",
                        "accuracy": accuracy_score(y_test, no_scaling_pred),
                        "macro_F1": f1_score(y_test, no_scaling_pred, average="macro"),
                    },
                ]
            )
            display(scaling_comparison)
            """
        ),
        md(
            """
            ## Демонстрация утечки данных

            > **Внимание. АНТИПРИМЕР - НЕ ИСПОЛЬЗОВАТЬ КАК РАБОЧУЮ МОДЕЛЬ.**
            > В следующей ячейке в признаки добавляется `risk_score`.
            > Этот столбец рассчитан из скрытого правила генератора и
            > частично кодирует диагностический класс. Его использование
            > завышает качество модели и нарушает учебную постановку.
            """
        ),
        code(
            """
            leakage_df = df.merge(diagnostics_df[["sample_id", "risk_score"]], on="sample_id", validate="one_to_one")
            X_leak = leakage_df[pd_features + ["risk_score"]]
            X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
                X_leak,
                y,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            leakage_model = RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            leakage_model.fit(X_train_leak, y_train_leak)
            leakage_pred = leakage_model.predict(X_test_leak)
            print("F1-macro строгой лучшей модели:", round(float(metrics_df.iloc[0]["f1_macro"]), 4))
            print("F1-macro модели с утечкой:", round(f1_score(y_test_leak, leakage_pred, average="macro"), 4))
            """
        ),
        code(
            todo_text(
                teacher,
                "risk_score является утечкой, потому что это производная диагностическая оценка, построенная с учетом скрытого правила разметки и близкая к целевому классу.",
                "pd_leakage_explanation",
                "объясните, почему risk_score нельзя использовать как базовый признак",
            )
        ),
        md("## Самостоятельный эксперимент"),
        code(
            todo_value(teacher, 3.0, "experiment_c", "рекомендуемый диапазон C для SVM: 0.1..10.0")
            + dedent(
                """
            experiment_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", C=experiment_c, gamma="scale", class_weight="balanced")),
                ]
            )
            experiment_model.fit(X_train, y_train)
            experiment_pred = experiment_model.predict(X_test)
            print("Accuracy:", round(accuracy_score(y_test, experiment_pred), 4))
            print("Recall macro:", round(recall_score(y_test, experiment_pred, average="macro"), 4))
            print("F1 macro:", round(f1_score(y_test, experiment_pred, average="macro"), 4))
            """
            )
        ),
        *dataset_assignment_cells_04_06(5),
        md(
            """
            ## Мини-задание по открытому источнику

            Mendeley Data `Partial Discharge Signals in Insulated Power
            Cables with Time-of-Arrival Annotations` содержит более близкую к
            реальной задаче форму данных: временные сигналы и аннотации
            импульсов. Для расширения работы укажите:

            1. какие признаки нужно извлечь из сырых сигналов перед
               классификацией;
            2. зачем фиксировать частоту дискретизации;
            3. чем PRPD-признаки отличаются от сырых временных отсчетов;
            4. какие риски утечки появляются, если импульсы одного
               измерительного опыта случайно перемешать между train и test.
            """
        ),
        md(
            """
            ## Задание для отчета

            1. Опишите физический смысл PRPD-представления.
            2. Сравните логистическую регрессию, дерево решений, случайный лес
               и два варианта SVM.
            3. Объясните, почему масштабирование признаков обязательно для SVM.
            4. Проанализируйте матрицу ошибок: какие классы чаще путаются.
            5. Укажите, почему `risk_score` является антипримером утечки.
            6. Выполните мини-задание по открытому источнику Mendeley на
               уровне постановки задачи без скачивания архива.

            Открытые источники для расширения: Mendeley Data `Partial
            Discharge Signals in Insulated Power Cables with Time-of-Arrival
            Annotations` и наборы PRPD-изображений частичных разрядов. Для
            реальных сигналов необходимо отдельно описывать частоту
            дискретизации, способ фильтрации и протокол аннотирования.
            """
        ),
    ]
    return cells


def build_practice_06(teacher: bool) -> list[nbformat.NotebookNode]:
    features = [
        "speed_rpm",
        "torque_nm",
        "current_a",
        "voltage_v",
        "temperature_c",
        "vibration_rms_mm_s",
        "acoustic_db",
        "cooling_flow_lpm",
        "efficiency",
        "pressure_kpa",
    ]
    cells = [
        md(
            """
            # Практическое занятие 6. Кластеризация режимов оборудования

            Цель занятия - выполнить обучение без учителя для группировки
            режимов оборудования по сенсорным признакам, сравнить k-means,
            DBSCAN и Gaussian Mixture, а затем интерпретировать найденные
            группы через диагностическую разметку.

            Обучение без учителя (unsupervised learning) означает, что модель
            не получает целевую переменную во время обучения. Диагностические
            метки используются только после кластеризации для проверки
            инженерной интерпретации.

            **Задача студента.** Нужно выбрать только сенсорные признаки,
            выполнить масштабирование, подобрать число кластеров, сравнить
            алгоритмы и дать инженерное описание найденных групп. Диагностическую
            разметку можно использовать только после обучения, иначе задача
            перестает быть кластеризацией.
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_06_equipment_modes_features.csv",
                "practice_06_equipment_modes_diagnostics.csv",
                "practice_04_06_dataset_catalog.csv",
                "practice_04_06_dataset_assignments.csv",
            ]
        ),
        code(COMMON_IMPORTS),
        source_section_04_06(),
        md(
            """
            ## Теоретический блок

            Кластеризация (clustering) - группировка объектов по сходству.
            Метод k-средних (k-means) ищет центры кластеров и минимизирует
            сумму квадратов расстояний до ближайшего центра.

            DBSCAN (Density-Based Spatial Clustering of Applications with
            Noise) - плотностной алгоритм, который выделяет плотные области и
            помечает разреженные точки как шум. Gaussian Mixture Model (GMM),
            гауссова смесь, описывает данные как смесь нескольких нормальных
            распределений.

            PCA (Principal Component Analysis), метод главных компонент,
            используется для визуализации многомерных данных на плоскости.
            """
        ),
        md(
            """
            ## Последовательность работы

            Pipeline кластеризации:

            1. загрузить feature-CSV без целевой переменной;
            2. выбрать только сенсорные признаки;
            3. масштабировать признаки;
            4. построить PCA-проекцию;
            5. подобрать число кластеров;
            6. сравнить k-means, DBSCAN и Gaussian Mixture;
            7. интерпретировать кластеры через средние профили признаков;
            8. только после этого присоединить diagnostics-CSV.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(12, 3.8))
            ax.axis("off")
            steps = [
                ("Сенсорные\\nпризнаки", 0.08),
                ("Масштабирование", 0.27),
                ("PCA\\nвизуализация", 0.46),
                ("Кластеризация", 0.65),
                ("Интерпретация\\nчерез diagnostics", 0.86),
            ]
            for text, x in steps:
                ax.text(
                    x,
                    0.55,
                    text,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f7f4", edgecolor="#9c755f"),
                    transform=ax.transAxes,
                )
            for x0, x1 in [(0.17, 0.22), (0.36, 0.41), (0.55, 0.60), (0.74, 0.81)]:
                ax.annotate(
                    "",
                    xy=(x1, 0.55),
                    xytext=(x0, 0.55),
                    xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
                )
            ax.set_title("Pipeline кластеризации: diagnostics подключается только после обучения")
            plt.show()
            """
        ),
        code(
            """
            FEATURES_FILE = DATA_DIR / "practice_06_equipment_modes_features.csv"
            DIAGNOSTICS_FILE = DATA_DIR / "practice_06_equipment_modes_diagnostics.csv"

            df = pd.read_csv(FEATURES_FILE)
            diagnostics_df = pd.read_csv(DIAGNOSTICS_FILE)
            full_df = df.merge(diagnostics_df, on="sample_id", validate="one_to_one")

            display(df.head())
            print("Размер feature-таблицы:", df.shape)
            print("Размер diagnostics-таблицы:", diagnostics_df.shape)
            """
        ),
        md(
            """
            ## Структура данных

            В feature-CSV нет целевой переменной. Столбцы `true_mode_label`,
            `mode_id`, `anomaly_flag` и `health_score` находятся только в
            diagnostics-CSV. Их нельзя использовать в обучении кластеризации,
            иначе задача обучения без учителя превращается в скрытую
            классификацию.
            """
        ),
        md(
            """
            На этом этапе студент должен убедиться, что feature-CSV содержит
            только наблюдаемые сенсорные признаки. Скрытая разметка режимов
            находится в diagnostics-CSV и будет использована только для
            проверки интерпретации кластеров.
            """
        ),
        code(
            """
            display(df.describe().T)
            print("Пропуски:")
            display(df.isna().sum().to_frame("missing_count"))

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(df["temperature_c"], bins=28, color="#4c78a8", edgecolor="white")
            axes[0].set_title("Температура")
            axes[0].set_xlabel("deg_C")

            axes[1].hist(df["vibration_rms_mm_s"], bins=28, color="#e15759", edgecolor="white")
            axes[1].set_title("Вибрация RMS")
            axes[1].set_xlabel("mm/s")

            axes[2].scatter(df["current_a"], df["temperature_c"], s=18, alpha=0.60)
            axes[2].set_title("Ток и температура")
            axes[2].set_xlabel("A")
            axes[2].set_ylabel("deg_C")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            Корреляционная матрица помогает увидеть группы связанных
            признаков. Сильная корреляция не означает причинность, но
            показывает, какие величины могут совместно определять расстояния
            между объектами при кластеризации.
            """
        ),
        code(
            """
            corr = df.select_dtypes(include=[np.number]).drop(columns=["sample_id"]).corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im, ax=ax, label="Коэффициент корреляции")
            ax.set_title("Тепловая карта корреляций сенсорных признаков")
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Выбор признаков и масштабирование"),
        code(
            todo_list(
                teacher,
                features,
                "cluster_features",
                "используйте только сенсорные признаки; не добавляйте true_mode_label и anomaly_flag",
            )
            + dedent(
                """
            forbidden_columns = {"true_mode_label", "mode_id", "anomaly_flag", "health_score", "maintenance_priority"}
            leaked = forbidden_columns.intersection(cluster_features)
            if leaked:
                raise ValueError(f"Обнаружена утечка диагностической разметки: {sorted(leaked)}")

            X = df[cluster_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print("Масштабированная матрица признаков:", X_scaled.shape)
            """
            )
        ),
        code(
            todo_text(
                teacher,
                "Diagnostics-CSV нельзя использовать до кластеризации, потому что он содержит скрытую разметку режимов и показатели состояния; их включение превратит обучение без учителя в скрытую классификацию.",
                "diagnostics_usage_explanation",
                "объясните, почему diagnostics-CSV нельзя использовать до кластеризации",
            )
        ),
        md("## PCA-визуализация"),
        code(
            """
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
            explained = pca.explained_variance_ratio_
            print("Доля объясненной дисперсии PC1 и PC2:", np.round(explained, 4))

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(pca_df["PC1"], pca_df["PC2"], s=18, alpha=0.65)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Проекция данных на две главные компоненты")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## PCA biplot

            Biplot - совместная визуализация объектов и направлений признаков
            в координатах главных компонент. Стрелки показывают, какие
            признаки сильнее связаны с направлениями PC1 и PC2. Это
            объяснительная схема, а не строгая физическая модель.
            """
        ),
        code(
            """
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=["PC1_loading", "PC2_loading"],
                index=cluster_features,
            )
            top_loading_features = (
                loadings.assign(length=lambda data: np.sqrt(data["PC1_loading"] ** 2 + data["PC2_loading"] ** 2))
                .sort_values("length", ascending=False)
                .head(6)
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(pca_df["PC1"], pca_df["PC2"], s=18, alpha=0.45)
            scale = 4.0
            for feature, row in top_loading_features.iterrows():
                ax.arrow(
                    0,
                    0,
                    row["PC1_loading"] * scale,
                    row["PC2_loading"] * scale,
                    color="#e15759",
                    head_width=0.08,
                    length_includes_head=True,
                )
                ax.text(row["PC1_loading"] * scale * 1.08, row["PC2_loading"] * scale * 1.08, feature, color="#e15759")
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA biplot: объекты и направления признаков")
            plt.tight_layout()
            plt.show()
            display(top_loading_features)
            """
        ),
        md("## Выбор числа кластеров для k-means"),
        code(
            """
            k_rows = []
            for k in range(2, 8):
                model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
                labels = model.fit_predict(X_scaled)
                k_rows.append(
                    {
                        "k": k,
                        "inertia": model.inertia_,
                        "silhouette": silhouette_score(X_scaled, labels),
                    }
                )
            k_metrics_df = pd.DataFrame(k_rows)
            display(k_metrics_df)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(k_metrics_df["k"], k_metrics_df["inertia"], marker="o")
            axes[0].set_title("Метод локтя")
            axes[0].set_xlabel("k")
            axes[0].set_ylabel("Inertia")

            axes[1].plot(k_metrics_df["k"], k_metrics_df["silhouette"], marker="o", color="#59a14f")
            axes[1].set_title("Силуэтный коэффициент")
            axes[1].set_xlabel("k")
            axes[1].set_ylabel("Silhouette")
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Обучение k-means, DBSCAN и Gaussian Mixture"),
        code(
            todo_value(teacher, 5, "n_clusters", "рекомендуемый диапазон числа кластеров: 3..6")
            + todo_value(teacher, 0.95, "dbscan_eps", "рекомендуемый диапазон eps для DBSCAN: 0.6..1.4")
            + dedent(
                """
            kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=RANDOM_STATE)
            kmeans_labels = kmeans.fit_predict(X_scaled)

            dbscan = DBSCAN(eps=dbscan_eps, min_samples=8)
            dbscan_labels = dbscan.fit_predict(X_scaled)

            gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=RANDOM_STATE)
            gmm_labels = gmm.fit_predict(X_scaled)

            def safe_silhouette(labels: np.ndarray) -> float:
                unique = set(labels)
                if len(unique) <= 1 or len(unique) >= len(labels):
                    return float("nan")
                return silhouette_score(X_scaled, labels)

            clustering_metrics = pd.DataFrame(
                [
                    {"method": "kmeans", "clusters": len(set(kmeans_labels)), "silhouette": safe_silhouette(kmeans_labels)},
                    {"method": "dbscan", "clusters": len(set(dbscan_labels)) - int(-1 in set(dbscan_labels)), "silhouette": safe_silhouette(dbscan_labels)},
                    {"method": "gaussian_mixture", "clusters": len(set(gmm_labels)), "silhouette": safe_silhouette(gmm_labels)},
                ]
            )
            display(clustering_metrics)
            """
            )
        ),
        code(
            todo_text(
                teacher,
                "Рациональное число кластеров выбирается не только по максимуму силуэтного коэффициента, но и по инженерной интерпретируемости профилей режимов.",
                "cluster_number_explanation",
                "обоснуйте выбранное число кластеров по силуэту и инженерной интерпретации",
            )
        ),
        md(
            """
            DBSCAN может присвоить части наблюдений метку `-1`. Это не номер
            кластера, а обозначение шума: точки недостаточно плотно связаны с
            соседями при выбранном `eps`.
            """
        ),
        code(
            """
            dbscan_noise_count = int((dbscan_labels == -1).sum())
            dbscan_summary = pd.Series(dbscan_labels).value_counts().sort_index().to_frame("count")
            display(dbscan_summary)
            print("Число точек DBSCAN с меткой -1 (шум):", dbscan_noise_count)
            """
        ),
        code(
            """
            cluster_plot_df = pca_df.copy()
            cluster_plot_df["kmeans_cluster"] = kmeans_labels
            cluster_plot_df["dbscan_cluster"] = dbscan_labels
            cluster_plot_df["gmm_cluster"] = gmm_labels

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax, column, title in zip(
                axes,
                ["kmeans_cluster", "dbscan_cluster", "gmm_cluster"],
                ["k-means", "DBSCAN", "Gaussian Mixture"],
            ):
                scatter = ax.scatter(cluster_plot_df["PC1"], cluster_plot_df["PC2"], c=cluster_plot_df[column], cmap="tab10", s=18, alpha=0.70)
                ax.set_title(title)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Интерпретация кластеров"),
        code(
            """
            interpreted_df = full_df.copy()
            interpreted_df["kmeans_cluster"] = kmeans_labels

            cluster_profile = interpreted_df.groupby("kmeans_cluster")[cluster_features].mean().round(3)
            display(cluster_profile)

            diagnostic_crosstab = pd.crosstab(
                interpreted_df["kmeans_cluster"],
                interpreted_df["true_mode_label"],
                normalize="index",
            ).round(3)
            display(diagnostic_crosstab)

            print("Adjusted Rand Index с диагностической разметкой:",
                  round(adjusted_rand_score(interpreted_df["true_mode_label"], kmeans_labels), 4))
            """
        ),
        md(
            """
            ## Профили кластеров

            Для инженерной интерпретации удобно рассматривать не только
            расположение точек на PCA-графике, но и средние нормированные
            значения признаков внутри каждого кластера. Такой профиль
            помогает назвать кластер: например, "высокая нагрузка",
            "повышенная вибрация" или "ухудшенное охлаждение".
            """
        ),
        code(
            """
            scaled_feature_df = pd.DataFrame(X_scaled, columns=cluster_features, index=df.index)
            scaled_feature_df["kmeans_cluster"] = kmeans_labels
            cluster_profile_scaled = scaled_feature_df.groupby("kmeans_cluster")[cluster_features].mean()

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(cluster_profile_scaled, cmap="coolwarm", aspect="auto", vmin=-2.0, vmax=2.0)
            ax.set_xticks(range(len(cluster_features)))
            ax.set_xticklabels(cluster_features, rotation=90)
            ax.set_yticks(range(len(cluster_profile_scaled.index)))
            ax.set_yticklabels([f"cluster {i}" for i in cluster_profile_scaled.index])
            fig.colorbar(im, ax=ax, label="Среднее нормированное значение")
            ax.set_title("Профили кластеров по нормированным признакам")
            plt.tight_layout()
            plt.show()
            display(cluster_profile_scaled.round(3))
            """
        ),
        code(
            todo_text(
                teacher,
                "Например, кластер с высокой температурой и низким расходом охлаждения можно интерпретировать как режим ухудшенного охлаждения; кластер с высокой вибрацией - как режим вибрационной тревоги.",
                "cluster_profile_interpretation",
                "опишите 2-3 кластера на основе профилей признаков",
            )
        ),
        md(
            """
            ## Антипример: скрытая классификация вместо кластеризации

            > **Внимание. АНТИПРИМЕР - НЕ ИСПОЛЬЗОВАТЬ КАК РАБОЧИЙ ПОДХОД.**
            > Если добавить в признаки `mode_id`, `anomaly_flag` или
            > `health_score`, алгоритм получает диагностическую разметку и
            > перестает решать задачу обучения без учителя. Такой результат
            > нельзя считать кластеризацией сенсорных режимов.
            """
        ),
        code(
            """
            leakage_features = cluster_features + ["mode_id", "anomaly_flag"]
            leakage_df = full_df[leakage_features]
            leakage_scaled = StandardScaler().fit_transform(leakage_df)
            leakage_labels = KMeans(n_clusters=n_clusters, n_init=30, random_state=RANDOM_STATE).fit_predict(leakage_scaled)
            print("ARI строгой кластеризации:",
                  round(adjusted_rand_score(full_df["true_mode_label"], kmeans_labels), 4))
            print("ARI антипримера с диагностической разметкой:",
                  round(adjusted_rand_score(full_df["true_mode_label"], leakage_labels), 4))
            """
        ),
        *dataset_assignment_cells_04_06(6),
        md(
            """
            ## Мини-задание по открытому источнику

            UCI `AI4I 2020 Predictive Maintenance Dataset` содержит
            сенсорные признаки, служебные коды изделия и признаки отказов.
            Для расширения работы укажите:

            1. какие столбцы можно считать сенсорными признаками;
            2. какие столбцы являются служебными идентификаторами;
            3. какие столбцы являются диагностической разметкой и не должны
               использоваться до кластеризации;
            4. какие графики следует построить для первичной интерпретации
               кластеров;
            5. как использовать признаки отказов только после обучения
               алгоритма кластеризации.
            """
        ),
        md(
            """
            ## Задание для отчета

            1. Объясните, чем кластеризация отличается от классификации.
            2. Обоснуйте выбранное число кластеров по графикам `inertia` и
               силуэтного коэффициента.
            3. Сравните k-means, DBSCAN и Gaussian Mixture.
            4. Дайте инженерное описание 2-3 найденных кластеров.
            5. Объясните, почему diagnostics-CSV нельзя использовать до
               завершения кластеризации.

            Открытые источники для расширения: UCI AI4I 2020 Predictive
            Maintenance Dataset и NASA C-MAPSS. Для открытых источников
            необходимо отдельно фиксировать, какие столбцы являются
            признаками, какие - отказами, а какие - служебной диагностической
            разметкой.
            """
        ),
    ]
    return cells


def main() -> None:
    notebook_specs = [
        (
            "04_haps_thermal_modeling",
            build_practice_04,
        ),
        (
            "05_partial_discharge_classification",
            build_practice_05,
        ),
        (
            "06_equipment_modes_clustering",
            build_practice_06,
        ),
    ]
    for stem, builder in notebook_specs:
        save_notebook(builder(teacher=False), NOTEBOOKS_STUDENT / f"{stem}_student.ipynb")
        save_notebook(builder(teacher=True), NOTEBOOKS_TEACHER / f"{stem}_teacher.ipynb")
        print(f"Собран блокнот: {stem}")


if __name__ == "__main__":
    main()

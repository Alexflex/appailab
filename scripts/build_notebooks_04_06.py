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
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_04_haps_thermal_features.csv",
                "practice_04_haps_thermal_diagnostics.csv",
            ]
        ),
        code(COMMON_IMPORTS),
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
        md("## Обучение моделей машинного обучения"),
        code(
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
                    max_depth=8,
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
        md(
            """
            ## Задание для отчета

            1. Объясните, почему разбиение выполнено по профилям, а не
               случайным перемешиванием строк.
            2. Сравните физически мотивированную модель, Ridge-регрессию,
               случайный лес и градиентный бустинг.
            3. Укажите две наиболее важные величины для прогноза температуры.
            4. Объясните, почему важность признака в случайном лесе не равна
               строгой физической причинности.
            5. Опишите антипример утечки данных и не используйте его метрики
               как основной результат.

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
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_05_partial_discharge_features.csv",
                "practice_05_partial_discharge_diagnostics.csv",
            ]
        ),
        code(COMMON_IMPORTS),
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
        md(
            """
            ## Задание для отчета

            1. Опишите физический смысл PRPD-представления.
            2. Сравните логистическую регрессию, дерево решений, случайный лес
               и два варианта SVM.
            3. Объясните, почему масштабирование признаков обязательно для SVM.
            4. Проанализируйте матрицу ошибок: какие классы чаще путаются.
            5. Укажите, почему `risk_score` является антипримером утечки.

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
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_06_equipment_modes_features.csv",
                "practice_06_equipment_modes_diagnostics.csv",
            ]
        ),
        code(COMMON_IMPORTS),
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

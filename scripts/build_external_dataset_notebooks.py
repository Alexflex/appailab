"""Сборка расширенных Jupyter Notebook по внешним наборам данных.

Создаются 18 блокнотов:
3 внешних набора данных * 3 занятия * 2 версии (student/teacher).
Студенческие версии содержат TODO-ячейки и сохраняются без output.
Преподавательские версии содержат заполненные параметры; outputs добавляются
отдельным запуском nbconvert --execute --inplace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks" / "external"
STUDENT_DIR = NOTEBOOKS_ROOT / "student"
TEACHER_DIR = NOTEBOOKS_ROOT / "teacher"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    short_name: str
    title: str
    source_url: str
    dataset_description: str
    data_structure: str
    processing_notes: str
    features_file: str
    diagnostics_file: str
    metadata_file: str
    group_column: str
    time_column: str
    time_kind: str
    lesson01_columns: list[str]
    regression_target: str
    regression_features: list[str]
    regression_leakage_features: list[str]
    forbidden_regression_features: list[str]
    classification_target: str
    classification_features: list[str]
    classification_leakage_features: list[str]
    forbidden_classification_features: list[str]
    class_positive_meaning: str
    regression_target_meaning: str


DATASETS = [
    DatasetConfig(
        dataset_id="zenodo_motor_temperature",
        short_name="zenodo_motor_temp",
        title="ElectricMotorTemperature, Zenodo TSML Archive",
        source_url="https://zenodo.org/records/11235562",
        dataset_description=(
            "Открытый набор ElectricMotorTemperature представлен как многомерные "
            "временные ряды (multivariate time series), то есть несколько "
            "синхронных каналов измерений, наблюдаемых в течение короткого "
            "временного фрагмента. Учебная задача состоит в переходе от "
            "исходной временной формы к табличным агрегированным признакам."
        ),
        data_structure=(
            "Исходный формат .ts содержит фрагменты длиной 60 отсчетов. "
            "Физические имена каналов в архиве не заданы, поэтому в компактном "
            "CSV используются нейтральные имена channel_00, channel_01 и далее. "
            "Для каждого канала рассчитаны средние значения, стандартные "
            "отклонения, минимумы, максимумы и тренды."
        ),
        processing_notes=(
            "Главная методическая особенность - сохранение группового разбиения "
            "по profile_id. Случайное перемешивание фрагментов может создать "
            "завышенную оценку качества, если близкие по происхождению фрагменты "
            "попадут одновременно в обучение и тест."
        ),
        features_file="external/zenodo_motor_temperature_features.csv",
        diagnostics_file="external/zenodo_motor_temperature_diagnostics.csv",
        metadata_file="external/zenodo_motor_temperature_metadata.md",
        group_column="profile_id",
        time_column="time_index",
        time_kind="порядковый индекс фрагмента внутри искусственно сформированного блока; это не физическое время",
        lesson01_columns=[
            "channel_00_mean",
            "channel_01_mean",
            "channel_02_mean",
            "channel_03_mean",
            "channel_04_mean",
            "channel_05_mean",
            "target_temperature_c",
        ],
        regression_target="target_temperature_c",
        regression_features=[
            "channel_00_mean",
            "channel_01_mean",
            "channel_02_mean",
            "channel_03_mean",
            "channel_04_mean",
            "channel_05_mean",
            "channel_00_trend",
            "channel_01_trend",
        ],
        regression_leakage_features=[
            "channel_00_mean",
            "channel_01_mean",
            "channel_02_mean",
            "channel_03_mean",
            "channel_04_mean",
            "channel_05_mean",
            "channel_00_std",
            "channel_01_std",
            "channel_02_std",
            "channel_03_std",
        ],
        forbidden_regression_features=[
            "target_temperature_c",
            "is_allowed",
        ],
        classification_target="is_allowed",
        classification_features=[
            "channel_00_mean",
            "channel_01_mean",
            "channel_02_mean",
            "channel_03_mean",
            "channel_04_mean",
            "channel_05_mean",
            "channel_00_trend",
            "channel_01_trend",
        ],
        classification_leakage_features=[
            "channel_00_mean",
            "channel_01_mean",
            "channel_02_mean",
            "channel_03_mean",
            "channel_04_mean",
            "channel_05_mean",
            "target_temperature_c",
        ],
        forbidden_classification_features=[
            "target_temperature_c",
            "thermal_limit_target_temperature_c",
            "is_allowed",
        ],
        class_positive_meaning="целевая температура временного фрагмента не выше учебного порога",
        regression_target_meaning="целевая температура временного фрагмента электродвигателя",
    ),
    DatasetConfig(
        dataset_id="zenodo_pmsm_inverter_fault",
        short_name="zenodo_inverter",
        title="Zenodo PMSM inverter fault diagnosis",
        source_url="https://zenodo.org/records/14482932",
        dataset_description=(
            "Набор содержит измерения инвертора постоянно-магнитной синхронной "
            "машины (Permanent Magnet Synchronous Motor, PMSM) в нормальном "
            "режиме и при отказах. Он полезен для связи анализа данных с "
            "диагностикой силовой электроники."
        ),
        data_structure=(
            "Feature-CSV содержит режимные признаки: прокси-момент, напряжение "
            "звена постоянного тока, ток и среднюю температуру. Diagnostics-CSV "
            "хранит максимальную температуру полумоста, код отказа FDD, фазные "
            "токи, мощности и дополнительные диагностические разности."
        ),
        processing_notes=(
            "Код отказа и диагностические разности нельзя автоматически включать "
            "в базовую модель, поскольку они раскрывают исходную разметку или "
            "являются производными диагностическими величинами."
        ),
        features_file="external/zenodo_pmsm_inverter_fault_features.csv",
        diagnostics_file="external/zenodo_pmsm_inverter_fault_diagnostics.csv",
        metadata_file="external/zenodo_pmsm_inverter_fault_metadata.md",
        group_column="profile_id",
        time_column="time_index",
        time_kind="время измерения из исходного файла в условных секундах эксперимента",
        lesson01_columns=[
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "temperature_c",
            "max_bridge_temp_c",
        ],
        regression_target="max_bridge_temp_c",
        regression_features=[
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "temperature_c",
        ],
        regression_leakage_features=[
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "temperature_c",
            "T1",
            "T2",
            "T3",
        ],
        forbidden_regression_features=[
            "max_bridge_temp_c",
            "T1",
            "T2",
            "T3",
            "is_allowed",
        ],
        classification_target="is_allowed",
        classification_features=[
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "temperature_c",
        ],
        classification_leakage_features=[
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "temperature_c",
            "Current_Imbalance",
            "Temp_Diff_Max",
        ],
        forbidden_classification_features=[
            "fault_code",
            "fault_label",
            "Current_Imbalance",
            "Temp_Diff_Max",
            "is_allowed",
        ],
        class_positive_meaning="код FDD равен F0, то есть нормальная работа",
        regression_target_meaning="максимальная температура полумоста инвертора",
    ),
    DatasetConfig(
        dataset_id="mendeley_ev_powertrain_efficiency",
        short_name="mendeley_ev",
        title="Processed Data for EV Powertrain Efficiency, Mendeley Data",
        source_url="https://data.mendeley.com/datasets/kbwr2z8r3y",
        dataset_description=(
            "Набор описывает движение электрического транспортного средства и "
            "расчетную эффективность его электропривода. Он демонстрирует "
            "переход от траекторных данных к энергетическим показателям."
        ),
        data_structure=(
            "Feature-CSV содержит скорость транспортного средства, ускорение, "
            "уклон, скорость и момент электродвигателя. Diagnostics-CSV хранит "
            "дату, координаты, расчетные показатели эффективности, суммарную "
            "силу сопротивления, механическую мощность и порог классификации "
            "эффективности."
        ),
        processing_notes=(
            "Для этого источника особенно важна защита от прокси-утечки: "
            "drivetrain_efficiency близко связан с motor_efficiency и не должен "
            "использоваться как строгий признак регрессии целевого КПД двигателя."
        ),
        features_file="external/mendeley_ev_powertrain_efficiency_features.csv",
        diagnostics_file="external/mendeley_ev_powertrain_efficiency_diagnostics.csv",
        metadata_file="external/mendeley_ev_powertrain_efficiency_metadata.md",
        group_column="profile_id",
        time_column="time_index",
        time_kind="Unix-время в секундах, полученное из столбца DateTime",
        lesson01_columns=[
            "vehicle_speed_m_s",
            "acceleration_m_s2",
            "slope_rad",
            "motor_speed_rpm",
            "motor_torque_nm",
            "motor_efficiency",
            "drivetrain_efficiency",
        ],
        regression_target="motor_efficiency",
        regression_features=[
            "vehicle_speed_m_s",
            "acceleration_m_s2",
            "slope_rad",
            "motor_speed_rpm",
            "motor_torque_nm",
        ],
        regression_leakage_features=[
            "vehicle_speed_m_s",
            "acceleration_m_s2",
            "slope_rad",
            "motor_speed_rpm",
            "motor_torque_nm",
            "drivetrain_efficiency",
            "Powertrain_efficiency_gear_SG",
        ],
        forbidden_regression_features=[
            "drivetrain_efficiency",
            "motor_efficiency",
            "Powertrain_efficiency_gear_SG",
            "mechanical_power_w",
            "efficiency_limit",
            "is_allowed",
        ],
        classification_target="is_allowed",
        classification_features=[
            "vehicle_speed_m_s",
            "acceleration_m_s2",
            "slope_rad",
            "motor_speed_rpm",
            "motor_torque_nm",
        ],
        classification_leakage_features=[
            "vehicle_speed_m_s",
            "acceleration_m_s2",
            "slope_rad",
            "motor_speed_rpm",
            "motor_torque_nm",
            "motor_efficiency",
            "drivetrain_efficiency",
        ],
        forbidden_classification_features=[
            "drivetrain_efficiency",
            "motor_efficiency",
            "Powertrain_efficiency_gear_SG",
            "efficiency_limit",
            "class_label",
            "is_allowed",
        ],
        class_positive_meaning="КПД электропривода не ниже учебного порога",
        regression_target_meaning="расчетный КПД электродвигателя транспортного средства",
    ),
]


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def setup_cell(config: DatasetConfig) -> str:
    return f"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
    Path.cwd().parent.parent.parent,
]

PROJECT_ROOT = None
for candidate in candidate_roots:
    if (candidate / "data").exists() and (candidate / "src").exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise RuntimeError("Не найден корень проекта appai_lab.")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = DATA_DIR / "{config.features_file}"
DIAGNOSTICS_FILE = DATA_DIR / "{config.diagnostics_file}"
METADATA_FILE = DATA_DIR / "{config.metadata_file}"
FALLBACK_FEATURES_FILE = DATA_DIR / "practice_02_motor_efficiency_features.csv"
FALLBACK_DIAGNOSTICS_FILE = DATA_DIR / "practice_02_motor_efficiency_diagnostics.csv"

RANDOM_STATE = 20260507
GROUP_COLUMN = "{config.group_column}"
TIME_COLUMN = "{config.time_column}"
TIME_KIND = "{config.time_kind}"
DATASET_ID = "{config.dataset_id}"
DATASET_TITLE = "{config.title}"
ALLOW_BASE_FALLBACK = False

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11


def load_external_or_fallback():
    if FEATURES_FILE.exists() and DIAGNOSTICS_FILE.exists():
        features = pd.read_csv(FEATURES_FILE)
        diagnostics = pd.read_csv(DIAGNOSTICS_FILE)
        source_status = "external"
    else:
        if not ALLOW_BASE_FALLBACK:
            raise FileNotFoundError(
                "Не найдены подготовленные внешние CSV. Выполните "
                "`venv/bin/python scripts/prepare_external_datasets.py`. "
                "Для аварийной демонстрации можно вручную установить "
                "ALLOW_BASE_FALLBACK = True, но тогда выводы блокнота будут "
                "относиться к базовому синтетическому набору, а не к внешнему источнику."
            )
        print(
            "Предупреждение: используется базовый синтетический CSV. "
            "Содержательные выводы по внешнему источнику в этом режиме недействительны."
        )
        features = pd.read_csv(FALLBACK_FEATURES_FILE)
        diagnostics = pd.read_csv(FALLBACK_DIAGNOSTICS_FILE)
        source_status = "fallback_base"
    full = features.merge(diagnostics, on="sample_id", how="left", validate="one_to_one")
    return features, diagnostics, full, source_status


def plot_correlation_heatmap(data, columns, title):
    corr = data[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(columns)), max(5, 0.65 * len(columns))))
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


def group_holdout_split(data, group_column, test_share=0.25, target_column=None):
    groups = np.array(sorted(data[group_column].dropna().unique()))
    test_count = max(1, int(np.ceil(len(groups) * test_share)))
    test_groups = groups[-test_count:]
    test_mask = data[group_column].isin(test_groups)
    train_idx = data.index[~test_mask]
    test_idx = data.index[test_mask]

    if target_column is None or target_column not in data.columns or data[target_column].nunique() < 2:
        return train_idx, test_idx, test_groups

    def _has_two_classes(index):
        return data.loc[index, target_column].nunique() >= 2

    if _has_two_classes(train_idx) and _has_two_classes(test_idx):
        return train_idx, test_idx, test_groups

    selected_groups = []
    for class_value in sorted(data[target_column].dropna().unique()):
        class_groups = np.array(sorted(data.loc[data[target_column] == class_value, group_column].dropna().unique()))
        class_test_count = max(1, int(np.ceil(len(class_groups) * test_share)))
        selected_groups.extend(class_groups[-class_test_count:].tolist())
    test_groups = np.array(sorted(set(selected_groups)))
    test_mask = data[group_column].isin(test_groups)
    train_idx = data.index[~test_mask]
    test_idx = data.index[test_mask]
    if not (_has_two_classes(train_idx) and _has_two_classes(test_idx)):
        raise ValueError(
            "Групповое разбиение не содержит оба класса в обучающей и тестовой "
            "выборках. Проверьте profile_id или правило формирования is_allowed."
        )
    return train_idx, test_idx, test_groups


def validate_features(features, forbidden, context):
    leaked = set(features) & set(forbidden)
    if leaked:
        raise AssertionError(
            f"Обнаружена утечка данных в {{context}}: {{sorted(leaked)}}"
        )
    return list(features)


def existing_columns(data, columns):
    result = []
    for column in columns:
        if column in data.columns and column not in result:
            result.append(column)
    return result


def compact_numeric_profile(data, columns):
    selected = data[existing_columns(data, columns)].select_dtypes(include=[np.number])
    rows = []
    for column in selected.columns:
        series = selected[column]
        rows.append(
            dict(
                column=column,
                missing_count=int(series.isna().sum()),
                missing_share=float(series.isna().mean()),
                unique_count=int(series.nunique(dropna=True)),
                minimum=float(series.min()),
                q25=float(series.quantile(0.25)),
                median=float(series.median()),
                q75=float(series.quantile(0.75)),
                maximum=float(series.max()),
                mean=float(series.mean()),
                std=float(series.std()),
            )
        )
    return pd.DataFrame(rows).set_index("column")


def print_missing_interpretation(data):
    missing_share = data.isna().mean().sort_values(ascending=False)
    nonzero = missing_share[missing_share > 0]
    if nonzero.empty:
        print("Интерпретация: пропуски не обнаружены. Для учебной работы это означает, что можно сосредоточиться на диапазонах, группах и риске утечки данных.")
    else:
        print("Интерпретация: обнаружены пропуски. Перед моделированием необходимо выбрать способ обработки и объяснить его физический смысл.")
        print(nonzero.head(10).to_string())


def print_group_interpretation(data, group_column):
    if group_column not in data.columns:
        print("Групповой столбец отсутствует. Следует явно обосновать выбранный способ разбиения данных.")
        return
    group_counts = data[group_column].value_counts()
    print(f"Число групп: {{group_counts.shape[0]}}.")
    print(f"Минимальный размер группы: {{group_counts.min()}}.")
    print(f"Максимальный размер группы: {{group_counts.max()}}.")
    print(
        "Интерпретация: если наблюдения внутри одной группы близки по происхождению, "
        "то строки одной группы нельзя произвольно делить между обучением и тестом."
    )


def print_top_correlations(data, columns, target_column, top_n=5):
    candidate_columns = [column for column in columns if column != target_column] + [target_column]
    usable_columns = existing_columns(data, candidate_columns)
    if target_column not in usable_columns:
        print(f"Целевая переменная {{target_column}} отсутствует.")
        return pd.DataFrame()
    corr = data[usable_columns].corr(numeric_only=True)[target_column].drop(target_column, errors="ignore")
    result = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n).to_frame("correlation")
    print(
        "Интерпретация: высокая корреляция показывает сильную линейную связь, "
        "но не доказывает причинность. Также необходимо проверить, не является ли "
        "признак производным от целевой переменной."
    )
    return result


def interpret_regression_metrics(metrics):
    best_model = metrics["R2"].idxmax()
    best_r2 = metrics.loc[best_model, "R2"]
    best_mae = metrics.loc[best_model, "MAE"]
    print(f"Лучшая модель по R2: {{best_model}}.")
    print(f"R2 = {{best_r2:.4f}}; MAE = {{best_mae:.4f}}.")
    print(
        "Интерпретация: MAE показывает средний модуль ошибки в единицах целевой "
        "переменной. R2 показывает долю объясненной дисперсии относительно "
        "модели среднего значения. Высокий R2 должен дополнительно проверяться "
        "на утечку данных и устойчивость к способу разбиения."
    )


def interpret_split_comparison(metrics):
    if {{"group_holdout", "random_split"}}.issubset(metrics.index):
        delta = metrics.loc["random_split", "R2"] - metrics.loc["group_holdout", "R2"] if "R2" in metrics.columns else np.nan
        print(f"Разница random_split - group_holdout по R2: {{delta:.4f}}.")
        print(
            "Интерпретация: если случайное разбиение существенно лучше группового, "
            "это может указывать на информационную близость соседних наблюдений и "
            "завышение качества при случайном перемешивании."
        )
    else:
        print("Сравнение разбиений недоступно.")


def interpret_classification_metrics(metrics):
    if "strict_decision_tree" in metrics.index:
        row = metrics.loc["strict_decision_tree"]
        print(f"Accuracy строгого дерева: {{row['accuracy']:.4f}}.")
        print(f"Recall недопустимого класса: {{row['recall_not_allowed']:.4f}}.")
        print(f"Доля ложных разрешений: {{row['dangerous_false_allowed_rate']:.4f}}.")
        print(
            "Интерпретация: для инженерной безопасности recall недопустимого класса "
            "и доля ложных разрешений важнее, чем одна общая accuracy."
        )


def interpret_depth_metrics(metrics):
    best_depth = metrics["f1_not_allowed"].idxmax()
    print(f"Наилучшая глубина по F1 недопустимого класса: {{best_depth}}.")
    print(
        "Интерпретация: рост глубины увеличивает гибкость дерева, но может привести "
        "к переобучению. Следует выбирать не максимальную глубину, а глубину, которая "
        "дает приемлемый баланс качества и интерпретируемости."
    )


def plot_missingness(data, title):
    missing_share = data.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * len(missing_share))))
    ax.barh(missing_share.index[::-1], missing_share.values[::-1], color="#4c78a8")
    ax.set_xlabel("Доля пропусков")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_boxplots(data, columns, title):
    selected = existing_columns(data, columns)
    if not selected:
        print("Нет доступных столбцов для построения диаграмм размаха.")
        return
    fig, axes = plt.subplots(1, len(selected), figsize=(max(10, 2.2 * len(selected)), 4))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, selected):
        ax.boxplot(data[column].dropna(), vert=True)
        ax.set_title(column, rotation=20)
        ax.set_ylabel("Значение")
    plt.suptitle(title, y=1.04)
    plt.tight_layout()
    plt.show()


def plot_time_or_group_profile(data, value_column, title):
    if value_column not in data.columns:
        print(f"Столбец {{value_column}} отсутствует.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    if TIME_COLUMN in data.columns:
        sample = data.sort_values([GROUP_COLUMN, TIME_COLUMN]).head(800)
        axes[0].plot(sample[TIME_COLUMN], sample[value_column], marker=".", linestyle="none", alpha=0.55)
        axes[0].set_xlabel(f"{{TIME_COLUMN}}: {{TIME_KIND}}")
        axes[0].set_ylabel(value_column)
        axes[0].set_title("Фрагмент временного, порядкового или траекторного профиля")
    else:
        axes[0].set_axis_off()
    if GROUP_COLUMN in data.columns:
        group_summary = data.groupby(GROUP_COLUMN)[value_column].median().head(40)
        axes[1].bar(group_summary.index.astype(str), group_summary.values, color="#f58518")
        axes[1].set_xlabel(GROUP_COLUMN)
        axes[1].set_ylabel(f"Медиана {{value_column}}")
        axes[1].set_title("Медианное значение по группам")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].set_axis_off()
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()


def role_table(feature_columns, diagnostic_columns, regression_target, classification_target):
    rows = []
    all_columns = list(feature_columns) + [column for column in diagnostic_columns if column not in feature_columns]
    for column in all_columns:
        if column == "sample_id":
            role = "служебный идентификатор"
        elif column in {{GROUP_COLUMN, TIME_COLUMN}}:
            role = "служебный или группировочный столбец"
        elif column == regression_target:
            role = "целевая переменная регрессии"
        elif column == classification_target:
            role = "целевая переменная классификации"
        elif column in feature_columns:
            role = "доступный столбец feature-CSV"
        else:
            role = "диагностический столбец diagnostics-CSV"
        rows.append(dict(column=column, role=role))
    return pd.DataFrame(rows)


features_df, diagnostics_df, full_df, source_status = load_external_or_fallback()
print("Источник данных:", source_status)
print("Файл признаков:", FEATURES_FILE if source_status == "external" else FALLBACK_FEATURES_FILE)
print("Размер feature-таблицы:", features_df.shape)
features_df.head()
"""


def lesson01_cells(config: DatasetConfig, teacher: bool) -> list[nbf.NotebookNode]:
    columns = config.lesson01_columns
    todo = (
        f"analysis_columns = {columns!r}"
        if teacher
        else (
            "# Рекомендуемый базовый набор для первого запуска уже заполнен.\n"
            "# TODO: после выполнения блокнота измените 2-3 столбца и сравните выводы.\n"
            f"analysis_columns = {columns!r}"
        )
    )
    return [
        md(f"""
# Занятие 01. Первичный анализ внешнего набора данных: {config.title}

## Теоретический блок

Набор данных рассматривается как инженерная таблица наблюдений. Наблюдение -
одна строка, соответствующая измеренному или расчетному состоянию объекта.
Признак (feature) - входная величина анализа. Диагностический столбец -
величина, полезная для объяснения источника данных, но потенциально опасная
как вход модели из-за утечки данных (data leakage).

Источник: {config.source_url}.

## Описание источника

{config.dataset_description}

Структура данных: {config.data_structure}

Особенность обработки: {config.processing_notes}

## План анализа

1. Проверить, какие файлы фактически загружены и какая версия данных
   используется в блокноте.
2. Составить паспорт набора данных и таблицу ролей столбцов.
3. Выполнить аудит качества данных: пропуски, диапазоны, распределения,
   группировка по профилям или датам.
4. Построить несколько визуализаций, объясняющих структуру признаков и
   возможные риски утечки данных.
5. Сформулировать вывод о пригодности источника для регрессии и
   классификации.
"""),
        code(setup_cell(config)),
        code(f"""
metadata_text = METADATA_FILE.read_text(encoding="utf-8") if METADATA_FILE.exists() else "Метаданные не найдены."
print(metadata_text)
"""),
        code(f"""
dataset_passport = pd.DataFrame(
    {{
        "field": [
            "dataset_id",
            "title",
            "source_status",
            "rows_features",
            "columns_features",
            "group_column",
            "time_column",
            "regression_target",
            "classification_target",
        ],
        "value": [
            DATASET_ID,
            DATASET_TITLE,
            source_status,
            features_df.shape[0],
            features_df.shape[1],
            GROUP_COLUMN,
            TIME_COLUMN,
            "{config.regression_target}",
            "{config.classification_target}",
        ],
    }}
)
dataset_passport
"""),
        code("""
print(
    "Интерпретация паспорта: этот блок фиксирует не только размер таблицы, "
    "но и единицу наблюдения, групповой столбец и целевые переменные. "
    "Без такого паспорта трудно доказать, что последующее разбиение данных "
    "и выбор признаков выполнены корректно."
)
print_group_interpretation(features_df, GROUP_COLUMN)
"""),
        code(f"""
column_roles = role_table(
    features_df.columns.tolist(),
    diagnostics_df.columns.tolist(),
    {config.regression_target!r},
    {config.classification_target!r},
)
column_roles
"""),
        md("""
## Как читать таблицу ролей столбцов

Таблица ролей показывает, какие поля можно использовать как признаки, а какие
нужны только для диагностики. Служебный идентификатор `sample_id` не имеет
физического смысла режима. Групповой столбец применяется для корректного
разбиения. Целевые переменные не должны попадать во вход модели, иначе
возникает утечка данных (data leakage), то есть модель получает ответ или его
производную форму.
"""),
        md("""
## Аудит качества данных

Аудит качества данных включает проверку пропусков, диапазонов, числа
уникальных значений и устойчивости группировки. Для внешних источников это
особенно важно, поскольку исходные данные могли быть получены из разных
экспериментов, поездок, профилей или режимов отказа.
"""),
        code(f"""
{todo}
compact_numeric_profile(full_df, analysis_columns)
"""),
        code(f"""
missing_table = (
    full_df.isna().sum().rename("missing_count").to_frame()
    .assign(missing_share=lambda x: x["missing_count"] / len(full_df))
)
missing_table[missing_table["missing_count"] > 0]
"""),
        code("""
print_missing_interpretation(full_df)
"""),
        code("""
plot_missingness(full_df, "Доля пропусков во feature-CSV и diagnostics-CSV")
"""),
        code("""
numeric_columns = full_df.select_dtypes(include=[np.number]).columns.tolist()
shown_columns = [column for column in analysis_columns if column in numeric_columns]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for ax, column in zip(axes, shown_columns[:6]):
    ax.hist(full_df[column].dropna(), bins=30, color="#4c78a8", edgecolor="white")
    ax.set_title(column)
    ax.set_ylabel("Число наблюдений")
for ax in axes[len(shown_columns[:6]):]:
    ax.set_axis_off()
plt.suptitle("Распределения основных величин", y=1.02)
plt.tight_layout()
plt.show()
"""),
        code("""
plot_boxplots(full_df, analysis_columns[:6], "Диаграммы размаха выбранных признаков")
"""),
        code("""
if GROUP_COLUMN in features_df.columns:
    group_counts = features_df[GROUP_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(group_counts.index.astype(str), group_counts.values, color="#f58518")
    ax.set_xlabel(GROUP_COLUMN)
    ax.set_ylabel("Число наблюдений")
    ax.set_title("Распределение наблюдений по группам или профилям")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
plt.show()
group_counts.head(20) if GROUP_COLUMN in features_df.columns else "Групповой столбец отсутствует"
"""),
        code("""
print_group_interpretation(features_df, GROUP_COLUMN)
"""),
        code(f"""
plot_time_or_group_profile(
    full_df,
    {config.regression_target!r},
    "Профиль целевой величины по времени или группам",
)
"""),
        code("""
plot_correlation_heatmap(
    full_df,
    [column for column in analysis_columns if column in full_df.columns],
    "Корреляции выбранных столбцов",
)
"""),
        code(f"""
print_top_correlations(
    full_df,
    analysis_columns,
    {config.regression_target!r},
)
"""),
        code(f"""
target_columns = [{config.regression_target!r}, {config.classification_target!r}]
available_targets = existing_columns(full_df, target_columns)
fig, axes = plt.subplots(1, len(available_targets), figsize=(6 * len(available_targets), 4))
axes = np.atleast_1d(axes)
for ax, column in zip(axes, available_targets):
    if full_df[column].nunique(dropna=True) <= 10:
        counts = full_df[column].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#54a24b")
        ax.set_ylabel("Число наблюдений")
    else:
        ax.hist(full_df[column].dropna(), bins=30, color="#4c78a8", edgecolor="white")
        ax.set_ylabel("Число наблюдений")
    ax.set_xlabel(column)
    ax.set_title(f"Распределение {{column}}")
plt.tight_layout()
plt.show()
"""),
        code(f"""
x_column = analysis_columns[0]
y_column = "{config.regression_target}"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(full_df[x_column], full_df[y_column], alpha=0.55)
ax.set_xlabel(x_column)
ax.set_ylabel(y_column)
ax.set_title("Связь выбранного признака с целевой величиной")
plt.tight_layout()
plt.show()
"""),
        code("""
diagnostic_preview_columns = diagnostics_df.columns[: min(10, len(diagnostics_df.columns))].tolist()
diagnostics_df[diagnostic_preview_columns].head()
"""),
        md("""
## Интерпретация результатов первичного анализа

В отчете необходимо отделить факты от интерпретации. Фактами являются число
наблюдений, число пропусков, диапазоны признаков и вид распределений.
Интерпретация должна объяснять, какие особенности источника могут повлиять на
модель: временная зависимость, неодинаковая длина групп, расчетные признаки,
производные целевые переменные и диагностические столбцы.
"""),
        md("""
## Типовые ошибки интерпретации

1. Считать `sample_id` физическим признаком. Это служебный номер строки.
2. Делать вывод о причинности только по корреляции. Корреляция показывает
   совместное изменение, но не доказывает физический механизм.
3. Автоматически переносить diagnostics-CSV в признаки модели. В нем могут
   находиться производные величины и элементы разметки.
4. Игнорировать групповой или временной порядок наблюдений. Для временных и
   профильных данных это может привести к завышенной оценке качества модели.

## Контрольные вопросы и мини-задания

1. Что является единицей наблюдения в данном источнике?
2. Какие столбцы являются целевыми переменными для занятий 02 и 03?
3. Найдите два признака с наибольшей корреляцией с целевой переменной и
   объясните, почему корреляция не равна причинности.
4. Измените `analysis_columns`, добавив один диагностический столбец, и
   объясните, почему его нельзя автоматически считать допустимым признаком
   модели.
"""),
        md("""
## Задание

1. Опишите источник данных, объект наблюдения и единицу наблюдения.
2. Разделите столбцы на измеряемые, расчетные, служебные и диагностические.
3. Постройте не менее двух графиков и одну таблицу описательной статистики.
4. Объясните, какие столбцы могут привести к утечке данных в занятиях 02-03.
5. Сформулируйте индивидуальный инженерный вывод.
"""),
    ]


def lesson02_cells(config: DatasetConfig, teacher: bool) -> list[nbf.NotebookNode]:
    features = config.regression_features
    leakage = config.regression_leakage_features
    forbidden = config.forbidden_regression_features
    feature_cell = (
        f"regression_features = {features!r}\nregression_target = {config.regression_target!r}"
        if teacher
        else (
            "# Рекомендуемый строгий набор признаков уже заполнен для воспроизводимого первого запуска.\n"
            "# TODO: выполните блокнот, затем измените один признак и сравните метрики.\n"
            f"regression_features = {features!r}\n"
            f"regression_target = {config.regression_target!r}"
        )
    )
    experiment_cell = (
        "experiment_degree = 2\nexperiment_alpha = 1.0"
        if teacher
        else (
            "# TODO: измените degree и alpha после базового запуска.\n"
            "experiment_degree = 2\n"
            "experiment_alpha = 1.0"
        )
    )
    return [
        md(f"""
# Занятие 02. Регрессия по внешнему набору данных: {config.title}

## Теоретический блок

Регрессия (regression) - задача прогнозирования непрерывной величины.
В данном блокноте прогнозируется {config.regression_target_meaning}. Для
временных или профильных инженерных данных особенно важен способ проверки:
случайное перемешивание соседних точек может завысить качество модели.
Поэтому используется holdout-разбиение по группам, то есть часть профилей
полностью откладывается в тестовую выборку.

## Описание источника и учебная гипотеза

{config.dataset_description}

Структура данных: {config.data_structure}

Особенность обработки: {config.processing_notes}

Учебная гипотеза занятия: строгий набор признаков должен давать
интерпретируемый, но не обязательно максимальный прогноз. Если качество резко
возрастает после добавления диагностических столбцов, это рассматривается как
признак возможной утечки данных, а не как автоматическое улучшение модели.

## Метрики регрессии

Средняя абсолютная ошибка (Mean Absolute Error, MAE) вычисляется как

$$MAE = \\frac{{1}}{{n}}\\sum_{{i=1}}^{{n}}|y_i-\\hat{{y}}_i|.$$

Корень из средней квадратичной ошибки (Root Mean Squared Error, RMSE)
вычисляется как

$$RMSE = \\sqrt{{\\frac{{1}}{{n}}\\sum_{{i=1}}^{{n}}(y_i-\\hat{{y}}_i)^2}}.$$

Коэффициент детерминации (coefficient of determination, R2) определяется как

$$R^2 = 1 - \\frac{{\\sum_{{i=1}}^{{n}}(y_i-\\hat{{y}}_i)^2}}{{\\sum_{{i=1}}^{{n}}(y_i-\\bar{{y}})^2}}.$$

MAE и RMSE выражаются в единицах целевой переменной. R2 является безразмерной
метрикой и показывает, насколько модель лучше базового прогноза средним
значением.
"""),
code(setup_cell(config)),
        code(f"""
{feature_cell}
forbidden_regression_features = set({forbidden!r})
regression_features = validate_features(
    regression_features,
    forbidden_regression_features,
    "regression_features",
)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

model_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=regression_features + [regression_target]
).copy()
train_idx, test_idx, test_groups = group_holdout_split(model_df, GROUP_COLUMN, test_share=0.25)
X_train = model_df.loc[train_idx, regression_features]
X_test = model_df.loc[test_idx, regression_features]
y_train = model_df.loc[train_idx, regression_target]
y_test = model_df.loc[test_idx, regression_target]

print("Тестовые группы:", test_groups)
print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
"""),
        code(f"""
regression_design_table = pd.DataFrame(
    {{
        "column": regression_features + [regression_target],
        "role": ["строгий признак"] * len(regression_features) + ["целевая переменная"],
    }}
)
forbidden_table = pd.DataFrame(
    {{
        "column": sorted(forbidden_regression_features),
        "role": "запрещено в строгой модели из-за риска утечки",
    }}
)
pd.concat([regression_design_table, forbidden_table], ignore_index=True)
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(y_train, bins=30, alpha=0.65, label="обучающая выборка", color="#4c78a8")
ax.hist(y_test, bins=30, alpha=0.65, label="тестовая выборка", color="#f58518")
ax.set_xlabel(regression_target)
ax.set_ylabel("Число наблюдений")
ax.set_title("Сравнение распределений целевой переменной")
ax.legend()
plt.tight_layout()
plt.show()
"""),
        code("""
target_correlations = (
    model_df[regression_features + [regression_target]]
    .corr(numeric_only=True)[regression_target]
    .drop(regression_target)
    .sort_values(key=lambda series: series.abs(), ascending=False)
    .to_frame("correlation_with_target")
)
target_correlations
"""),
        code("""
shown_features = regression_features[: min(4, len(regression_features))]
fig, axes = plt.subplots(1, len(shown_features), figsize=(5 * len(shown_features), 4))
axes = np.atleast_1d(axes)
for ax, column in zip(axes, shown_features):
    ax.scatter(model_df[column], model_df[regression_target], alpha=0.45)
    ax.set_xlabel(column)
    ax.set_ylabel(regression_target)
    ax.set_title(f"{column} и цель")
plt.tight_layout()
plt.show()
"""),
        code("""
plot_time_or_group_profile(
    model_df,
    regression_target,
    "Проверка целевой переменной по времени или группам",
)
"""),
        code("""
def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }

models = {
    "mean_baseline": None,
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "polynomial_ridge": Pipeline([
        ("input_scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("poly_scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
}

predictions = {}
rows = []
for name, model in models.items():
    if model is None:
        y_pred = np.full(len(y_test), y_train.mean())
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    predictions[name] = y_pred
    rows.append({"model": name, **regression_metrics(y_test, y_pred)})

metrics_df = pd.DataFrame(rows).set_index("model")
metrics_df
"""),
        code("""
interpret_regression_metrics(metrics_df)
"""),
        code("""
linear_pipeline = models["linear_regression"]
linear_coefficients = pd.Series(
    linear_pipeline.named_steps["model"].coef_,
    index=regression_features,
    name="standardized_coefficient",
).sort_values(key=lambda series: series.abs(), ascending=False)
linear_coefficients.to_frame()
"""),
        code("""
random_train_idx, random_test_idx = train_test_split(
    model_df.index,
    test_size=0.25,
    random_state=RANDOM_STATE,
)
random_model = Pipeline([
    ("input_scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("poly_scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
random_model.fit(model_df.loc[random_train_idx, regression_features], model_df.loc[random_train_idx, regression_target])
random_pred = random_model.predict(model_df.loc[random_test_idx, regression_features])

split_comparison = pd.DataFrame(
    [
        {"split_type": "group_holdout", **regression_metrics(y_test, predictions["polynomial_ridge"])},
        {
            "split_type": "random_split",
            **regression_metrics(model_df.loc[random_test_idx, regression_target], random_pred),
        },
    ]
).set_index("split_type")
split_comparison
"""),
        code("""
interpret_split_comparison(split_comparison)
"""),
        code("""
selected_pred = predictions["polynomial_ridge"]
residuals = y_test - selected_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_test, selected_pred, alpha=0.65)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black")
axes[0].set_xlabel("Фактическое значение")
axes[0].set_ylabel("Прогноз")
axes[0].set_title("Фактические и прогнозные значения")
axes[1].scatter(selected_pred, residuals, alpha=0.65, color="#f58518")
axes[1].axhline(0.0, color="black")
axes[1].set_xlabel("Прогноз")
axes[1].set_ylabel("Остаток")
axes[1].set_title("График остатков")
plt.tight_layout()
plt.show()
"""),
        code("""
residual_df = pd.DataFrame(
    {
        "sample_id": model_df.loc[y_test.index, "sample_id"],
        GROUP_COLUMN: model_df.loc[y_test.index, GROUP_COLUMN],
        "y_true": y_test,
        "y_pred": selected_pred,
        "residual": residuals,
        "abs_error": np.abs(residuals),
    }
)
residual_df["target_bin"] = pd.qcut(residual_df["y_true"], q=4, duplicates="drop")
error_by_target_bin = residual_df.groupby("target_bin", observed=True).agg(
    count=("abs_error", "size"),
    mean_abs_error=("abs_error", "mean"),
    mean_residual=("residual", "mean"),
)
error_by_target_bin
"""),
        code("""
print(
    "Интерпретация: если средняя абсолютная ошибка заметно различается по "
    "квартилям целевой переменной, модель неодинаково точна в разных "
    "режимах работы объекта. Это важнее, чем одна усредненная метрика."
)
"""),
        code("""
group_error = (
    residual_df.groupby(GROUP_COLUMN)
    .agg(
        count=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
        max_abs_error=("abs_error", "max"),
    )
    .sort_values("mean_abs_error", ascending=False)
    .head(10)
)
group_error
"""),
        code("""
print(
    "Интерпретация: группы с наибольшей ошибкой следует проверить отдельно. "
    "Они могут соответствовать другому режиму испытаний, поездки или отказа."
)
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(residual_df["residual"], bins=30, color="#4c78a8", edgecolor="white")
axes[0].axvline(0.0, color="black")
axes[0].set_xlabel("Остаток")
axes[0].set_ylabel("Число наблюдений")
axes[0].set_title("Распределение остатков")
axes[1].bar(error_by_target_bin.index.astype(str), error_by_target_bin["mean_abs_error"], color="#f58518")
axes[1].set_xlabel("Интервал фактической цели")
axes[1].set_ylabel("Средняя абсолютная ошибка")
axes[1].set_title("Ошибка по диапазонам целевой переменной")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
"""),
        code("""
degrees = [1, 2, 3]
alphas = [0.01, 0.1, 1.0, 10.0]
grid_rows = []
for degree in degrees:
    for alpha in alphas:
        grid_model = Pipeline([
            ("input_scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("poly_scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
        grid_model.fit(X_train, y_train)
        grid_pred = grid_model.predict(X_test)
        grid_rows.append(
            {
                "degree": degree,
                "alpha": alpha,
                **regression_metrics(y_test, grid_pred),
            }
        )

grid_metrics = pd.DataFrame(grid_rows)
grid_r2 = grid_metrics.pivot(index="degree", columns="alpha", values="R2")

fig, ax = plt.subplots(figsize=(7, 4))
image = ax.imshow(grid_r2.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(grid_r2.columns)))
ax.set_xticklabels(grid_r2.columns)
ax.set_yticks(range(len(grid_r2.index)))
ax.set_yticklabels(grid_r2.index)
ax.set_xlabel("alpha")
ax.set_ylabel("Степень полинома")
ax.set_title("R2 при разных degree и alpha")
fig.colorbar(image, ax=ax, label="R2")
for i in range(grid_r2.shape[0]):
    for j in range(grid_r2.shape[1]):
        ax.text(j, i, f"{grid_r2.iloc[i, j]:.2f}", ha="center", va="center", color="white")
plt.tight_layout()
plt.show()

grid_metrics.sort_values("R2", ascending=False).head(10)
"""),
        code("""
best_grid_row = grid_metrics.sort_values("R2", ascending=False).iloc[0]
print(
    "Интерпретация сетки экспериментов: лучший вариант в этой сетке имеет "
    f"degree={int(best_grid_row['degree'])}, alpha={best_grid_row['alpha']} "
    f"и R2={best_grid_row['R2']:.4f}."
)
print(
    "Если увеличение степени полинома почти не повышает качество, более простая "
    "модель предпочтительнее, поскольку ее легче объяснить и труднее переобучить."
)
"""),
        md("""
## Демонстрация риска утечки данных

Ниже используется расширенный набор признаков, включающий диагностические или
расчетные столбцы. Если метрики резко улучшаются, результат нельзя считать
доказательством качества базовой модели: возможно, модель получила
информацию, недоступную в момент реального прогноза.
"""),
        code(f"""
leakage_features = [column for column in {leakage!r} if column in full_df.columns]
leakage_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=leakage_features + [regression_target]
).copy()
if len(leakage_features) >= 2 and len(leakage_df) > 20:
    train_idx_l, test_idx_l, _ = group_holdout_split(leakage_df, GROUP_COLUMN, test_share=0.25)
    leakage_model = Pipeline([
        ("input_scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("poly_scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    leakage_model.fit(leakage_df.loc[train_idx_l, leakage_features], leakage_df.loc[train_idx_l, regression_target])
    leakage_pred = leakage_model.predict(leakage_df.loc[test_idx_l, leakage_features])
    leakage_metrics = pd.Series(
        regression_metrics(leakage_df.loc[test_idx_l, regression_target], leakage_pred),
        name="leakage_demo",
    )
else:
    leakage_metrics = pd.Series(dtype=float, name="leakage_demo")
leakage_metrics.to_frame("value")
"""),
        code("""
if not leakage_metrics.empty:
    strict_r2 = metrics_df.loc["polynomial_ridge", "R2"]
    leakage_r2 = leakage_metrics.get("R2", np.nan)
    print(f"R2 строгой модели: {strict_r2:.4f}.")
    print(f"R2 демонстрации утечки: {leakage_r2:.4f}.")
    print(
        "Интерпретация: резкий рост качества после добавления диагностических "
        "столбцов не следует трактовать как инженерный успех. Сначала нужно "
        "проверить, доступны ли эти признаки в момент реального прогноза."
    )
"""),
        code(f"""
{experiment_cell}
experiment_model = Pipeline([
    ("input_scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=experiment_degree, include_bias=False)),
    ("poly_scaler", StandardScaler()),
    ("model", Ridge(alpha=experiment_alpha)),
])
experiment_model.fit(X_train, y_train)
experiment_pred = experiment_model.predict(X_test)
pd.Series(regression_metrics(y_test, experiment_pred), name="experiment")
"""),
        md("""
## Интерпретация экспериментов

Сравнение group holdout и random split показывает, насколько оценка качества
зависит от процедуры разбиения. Если случайное разбиение дает заметно более
высокий результат, это может означать, что соседние или однородные фрагменты
попадают одновременно в обучение и тест. Анализ остатков по диапазонам цели
показывает, в каких режимах модель ошибается сильнее. Сетка `degree` и
`alpha` демонстрирует компромисс между гибкостью модели и регуляризацией.
"""),
        md("""
## Типовые ошибки интерпретации

1. Сравнивать модели только по R2 и не анализировать MAE, RMSE и остатки.
2. Считать случайное разбиение всегда корректным. Для временных и профильных
   данных оно может завышать качество.
3. Использовать диагностические столбцы как обычные признаки без проверки
   доступности в момент прогноза.
4. Выбирать наиболее сложную модель только потому, что она дала немного
   больший R2.

## Контрольные вопросы и мини-задания

1. Почему выбранная целевая переменная является непрерывной величиной?
2. Чем MAE отличается от RMSE с точки зрения чувствительности к крупным
   ошибкам?
3. Почему `group_holdout` может быть строже, чем `random_split`?
4. Измените `experiment_degree` и `experiment_alpha`, затем объясните, как
   изменились MAE, RMSE и R2.
5. Найдите группу с наибольшей ошибкой и сформулируйте инженерную гипотезу,
   почему она сложна для модели.
"""),
        md("""
## Задание

1. Укажите целевую переменную и строгий набор признаков.
2. Объясните, почему применено разбиение по профилям или датам.
3. Сравните базовую модель, линейную регрессию и полиномиальную гребневую
   регрессию (Ridge regression).
4. Постройте график остатков и объясните систематические ошибки.
5. Отдельно опишите демонстрацию утечки данных.
"""),
    ]


def lesson03_cells(config: DatasetConfig, teacher: bool) -> list[nbf.NotebookNode]:
    features = config.classification_features
    leakage = config.classification_leakage_features
    forbidden = config.forbidden_classification_features
    feature_cell = (
        f"classification_features = {features!r}\nclassification_target = {config.classification_target!r}"
        if teacher
        else (
            "# Рекомендуемый строгий набор признаков уже заполнен для воспроизводимого первого запуска.\n"
            "# TODO: после выполнения блокнота измените состав признаков и объясните изменение ошибок.\n"
            f"classification_features = {features!r}\n"
            f"classification_target = {config.classification_target!r}"
        )
    )
    depth_cell = (
        "experiment_depth = 3"
        if teacher
        else (
            "# TODO: измените глубину дерева и сравните recall недопустимого класса.\n"
            "experiment_depth = 3"
        )
    )
    return [
        md(f"""
# Занятие 03. Дерево решений по внешнему набору данных: {config.title}

## Теоретический блок

Классификация (classification) - задача отнесения наблюдения к одному из
заранее заданных классов. В данном блокноте целевая переменная
`{config.classification_target}` кодирует условно допустимый режим: {config.class_positive_meaning}.
Дерево решений (Decision Tree) используется как интерпретируемая модель,
поскольку его правила можно записать в виде пороговых инженерных условий.

## Описание источника и учебная гипотеза

{config.dataset_description}

Структура данных: {config.data_structure}

Особенность обработки: {config.processing_notes}

Учебная гипотеза занятия: дерево решений должно быть не только точным, но и
объяснимым. Поэтому оцениваются не только accuracy, но и матрица ошибок,
полнота критического класса, влияние глубины дерева, распределение ошибок по
диагностическим меткам и риск утечки данных.

## Метрики классификации

Матрица ошибок (confusion matrix) сопоставляет истинный класс и предсказанный
класс. Для бинарной классификации используются четыре величины: TP, TN, FP и
FN. Доля правильных ответов (accuracy) вычисляется как

$$accuracy = \\frac{{TP + TN}}{{TP + TN + FP + FN}}.$$

Полнота (recall) для выбранного класса показывает, какая доля объектов этого
класса найдена моделью:

$$recall = \\frac{{TP}}{{TP + FN}}.$$

Точность положительных предсказаний (precision) показывает, какая доля
предсказаний выбранного класса является правильной:

$$precision = \\frac{{TP}}{{TP + FP}}.$$

F1-мера (F1-score) является гармоническим средним precision и recall:

$$F1 = \\frac{{2 \\cdot precision \\cdot recall}}{{precision + recall}}.$$

В инженерной диагностике особое внимание уделяется ошибке, при которой
недопустимый или отказный режим распознан как допустимый.
"""),
code(setup_cell(config)),
        code(f"""
{feature_cell}
forbidden_classification_features = set({forbidden!r})
classification_features = validate_features(
    classification_features,
    forbidden_classification_features,
    "classification_features",
)

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

model_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=classification_features + [classification_target]
).copy()
train_idx, test_idx, test_groups = group_holdout_split(
    model_df,
    GROUP_COLUMN,
    test_share=0.25,
    target_column=classification_target,
)
X_train = model_df.loc[train_idx, classification_features]
X_test = model_df.loc[test_idx, classification_features]
y_train = model_df.loc[train_idx, classification_target].astype(int)
y_test = model_df.loc[test_idx, classification_target].astype(int)

print("Тестовые группы:", test_groups)
print("Распределение классов в полной выборке:")
print(model_df[classification_target].value_counts().sort_index())
"""),
        code(f"""
classification_design_table = pd.DataFrame(
    {{
        "column": classification_features + [classification_target],
        "role": ["строгий признак"] * len(classification_features) + ["целевая переменная"],
    }}
)
forbidden_table = pd.DataFrame(
    {{
        "column": sorted(forbidden_classification_features),
        "role": "запрещено в строгой модели из-за риска утечки",
    }}
)
pd.concat([classification_design_table, forbidden_table], ignore_index=True)
"""),
        code("""
class_share = model_df[classification_target].value_counts(normalize=True).sort_index()
print("Доли классов:")
print(class_share.to_string())
majority_share = class_share.max()
print(
    "Интерпретация: если доля большинства велика, accuracy базовой модели "
    f"может быть около {majority_share:.3f} даже без обнаружения критического класса."
)
"""),
        code("""
fig, ax = plt.subplots(figsize=(6, 4))
class_counts = model_df[classification_target].value_counts().sort_index()
ax.bar(class_counts.index.astype(str), class_counts.values, color=["#e45756", "#54a24b"][:len(class_counts)])
ax.set_xlabel("Класс")
ax.set_ylabel("Число наблюдений")
ax.set_title("Распределение классов")
plt.tight_layout()
plt.show()
"""),
        code("""
if GROUP_COLUMN in model_df.columns:
    class_by_group = (
        model_df.groupby([GROUP_COLUMN, classification_target])
        .size()
        .unstack(fill_value=0)
    )
    class_by_group_share = class_by_group.div(class_by_group.sum(axis=1), axis=0)
else:
    class_by_group_share = pd.DataFrame()

fig, ax = plt.subplots(figsize=(10, 4))
if not class_by_group_share.empty:
    class_by_group_share.head(40).plot(kind="bar", stacked=True, ax=ax, color=["#e45756", "#54a24b"])
    ax.set_ylabel("Доля класса внутри группы")
    ax.set_title("Распределение классов по группам")
    ax.tick_params(axis="x", rotation=45)
else:
    ax.text(0.5, 0.5, "Групповой столбец отсутствует", ha="center", va="center")
    ax.set_axis_off()
plt.tight_layout()
plt.show()

class_by_group.head(10) if GROUP_COLUMN in model_df.columns else "Групповой столбец отсутствует"
"""),
        code("""
scatter_features = classification_features[:2]
if len(scatter_features) >= 2:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = model_df[classification_target].map({0: "#e45756", 1: "#54a24b"})
    ax.scatter(
        model_df[scatter_features[0]],
        model_df[scatter_features[1]],
        c=colors,
        alpha=0.45,
    )
    ax.set_xlabel(scatter_features[0])
    ax.set_ylabel(scatter_features[1])
    ax.set_title("Два признака и целевой класс")
    plt.tight_layout()
    plt.show()
scatter_features
"""),
        code("""
def classification_metrics(y_true, y_pred):
    cm_local = confusion_matrix(y_true, y_pred, labels=[0, 1])
    not_allowed_total = cm_local[0].sum()
    dangerous_false_allowed_rate = cm_local[0, 1] / not_allowed_total if not_allowed_total > 0 else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_not_allowed": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_not_allowed": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_not_allowed": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "dangerous_false_allowed_rate": dangerous_false_allowed_rate,
    }

majority_class = int(y_train.mode().iloc[0])
baseline_pred = np.full(len(y_test), majority_class, dtype=int)

tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=RANDOM_STATE)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

metrics_df = pd.DataFrame(
    [
        {"model": "majority_baseline", **classification_metrics(y_test, baseline_pred)},
        {"model": "strict_decision_tree", **classification_metrics(y_test, y_pred)},
    ]
).set_index("model")
metrics_df
"""),
        code("""
interpret_classification_metrics(metrics_df)
"""),
        code("""
random_train_idx, random_test_idx = train_test_split(
    model_df.index,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=model_df[classification_target].astype(int),
)
random_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=RANDOM_STATE)
random_tree.fit(
    model_df.loc[random_train_idx, classification_features],
    model_df.loc[random_train_idx, classification_target].astype(int),
)
random_pred = random_tree.predict(model_df.loc[random_test_idx, classification_features])

split_metrics = pd.DataFrame(
    [
        {"split_type": "group_holdout", **classification_metrics(y_test, y_pred)},
        {
            "split_type": "random_split",
            **classification_metrics(
                model_df.loc[random_test_idx, classification_target].astype(int),
                random_pred,
            ),
        },
    ]
).set_index("split_type")
split_metrics
"""),
        code("""
if {"group_holdout", "random_split"}.issubset(split_metrics.index):
    delta_recall = (
        split_metrics.loc["random_split", "recall_not_allowed"]
        - split_metrics.loc["group_holdout", "recall_not_allowed"]
    )
    print(f"Разница random_split - group_holdout по recall недопустимого класса: {delta_recall:.4f}.")
    print(
        "Интерпретация: если случайное разбиение существенно лучше, необходимо "
        "проверить, не попали ли близкие наблюдения одной группы в обе части выборки."
    )
"""),
        code("""
depth_rows = []
for depth in range(1, 9):
    depth_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=8, random_state=RANDOM_STATE)
    depth_model.fit(X_train, y_train)
    depth_pred = depth_model.predict(X_test)
    depth_rows.append({"max_depth": depth, **classification_metrics(y_test, depth_pred)})

depth_metrics_df = pd.DataFrame(depth_rows).set_index("max_depth")

fig, ax = plt.subplots(figsize=(8, 4))
for metric in ["accuracy", "recall_not_allowed", "f1_not_allowed"]:
    ax.plot(depth_metrics_df.index, depth_metrics_df[metric], marker="o", label=metric)
ax.set_xlabel("Максимальная глубина дерева")
ax.set_ylabel("Значение метрики")
ax.set_title("Влияние глубины дерева на качество классификации")
ax.set_ylim(0.0, 1.05)
ax.legend()
plt.tight_layout()
plt.show()

depth_metrics_df
"""),
        code("""
interpret_depth_metrics(depth_metrics_df)
"""),
        code("""
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["недопустимый", "допустимый"])
disp.plot(values_format="d", cmap="Blues")
plt.title("Матрица ошибок строгого дерева решений")
plt.show()
"""),
        code("""
if hasattr(tree_model, "predict_proba"):
    allowed_probability = tree_model.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(allowed_probability[y_test == 0], bins=20, alpha=0.65, label="истинно недопустимые", color="#e45756")
    axes[0].hist(allowed_probability[y_test == 1], bins=20, alpha=0.65, label="истинно допустимые", color="#54a24b")
    axes[0].set_xlabel("Вероятность класса 1")
    axes[0].set_ylabel("Число наблюдений")
    axes[0].set_title("Распределение вероятностей дерева")
    axes[0].legend()

    threshold_rows = []
    for threshold in np.linspace(0.1, 0.9, 9):
        threshold_pred = (allowed_probability >= threshold).astype(int)
        threshold_rows.append({"threshold": threshold, **classification_metrics(y_test, threshold_pred)})
    threshold_metrics = pd.DataFrame(threshold_rows)
    axes[1].plot(threshold_metrics["threshold"], threshold_metrics["recall_not_allowed"], marker="o", label="recall_not_allowed")
    axes[1].plot(
        threshold_metrics["threshold"],
        threshold_metrics["dangerous_false_allowed_rate"],
        marker="o",
        label="dangerous_false_allowed_rate",
    )
    axes[1].set_xlabel("Порог вероятности класса 1")
    axes[1].set_ylabel("Значение")
    axes[1].set_title("Компромисс порога решения")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
else:
    threshold_metrics = pd.DataFrame()

threshold_metrics
"""),
        code("""
if not threshold_metrics.empty:
    best_threshold = threshold_metrics.sort_values(
        ["dangerous_false_allowed_rate", "recall_not_allowed"],
        ascending=[True, False],
    ).iloc[0]
    print(
        "Интерпретация порога решения: уменьшение доли ложных разрешений может "
        "потребовать изменения порога вероятности класса 1. В выбранной сетке "
        f"минимальная доля ложных разрешений достигается при threshold={best_threshold['threshold']:.2f}."
    )
"""),
        code("""
label_candidates = ["mode_label", "class_label", "fault_label", "fault_code"]
available_label_columns = [column for column in label_candidates if column in diagnostics_df.columns]

test_error_details = model_df.loc[test_idx, ["sample_id", classification_target]].copy()
test_error_details["true_class"] = y_test
test_error_details["predicted_class"] = y_pred
test_error_details["error_type"] = np.select(
    [
        (test_error_details["true_class"] == 0) & (test_error_details["predicted_class"] == 1),
        (test_error_details["true_class"] == 1) & (test_error_details["predicted_class"] == 0),
        test_error_details["true_class"] == test_error_details["predicted_class"],
    ],
    ["false_allowed", "false_blocked", "correct"],
    default="other",
)

if available_label_columns:
    test_error_details = test_error_details.merge(
        diagnostics_df[["sample_id"] + available_label_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    summary_column = available_label_columns[0]
    error_summary = (
        test_error_details.groupby([summary_column, "error_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
else:
    error_summary = test_error_details["error_type"].value_counts().to_frame("count")

error_summary
"""),
        code("""
print(
    "Интерпретация: сводка ошибок по диагностическим меткам показывает, какие "
    "физические или расчетные режимы модель путает чаще всего. Именно эти "
    "режимы следует обсуждать в инженерном выводе, а не только общую accuracy."
)
"""),
        code("""
test_error_details.sort_values("error_type").head(15)
"""),
        code("""
importance = pd.Series(tree_model.feature_importances_, index=classification_features).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
importance.plot(kind="barh", ax=ax, color="#4c78a8")
ax.set_xlabel("Относительная важность")
ax.set_title("Важность признаков строгого дерева решений")
plt.tight_layout()
plt.show()
importance.to_frame("importance")
"""),
        md("""
## Демонстрация риска утечки данных

Если в классификацию включить столбцы, из которых непосредственно построена
целевая переменная, дерево будет воспроизводить правило разметки, а не решать
независимую прогностическую задачу.
"""),
        code(f"""
leakage_features = [column for column in {leakage!r} if column in full_df.columns]
leakage_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=leakage_features + [classification_target]
).copy()
if len(leakage_features) >= 2 and leakage_df[classification_target].nunique() == 2:
    train_idx_l, test_idx_l, _ = group_holdout_split(
        leakage_df,
        GROUP_COLUMN,
        test_share=0.25,
        target_column=classification_target,
    )
    leakage_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=RANDOM_STATE)
    leakage_tree.fit(leakage_df.loc[train_idx_l, leakage_features], leakage_df.loc[train_idx_l, classification_target].astype(int))
    leakage_pred = leakage_tree.predict(leakage_df.loc[test_idx_l, leakage_features])
    leakage_metrics = pd.Series(
        classification_metrics(leakage_df.loc[test_idx_l, classification_target].astype(int), leakage_pred),
        name="leakage_demo",
    )
else:
    leakage_metrics = pd.Series(dtype=float, name="leakage_demo")
leakage_metrics.to_frame("value")
"""),
        code("""
if not leakage_metrics.empty:
    strict_accuracy = metrics_df.loc["strict_decision_tree", "accuracy"]
    leakage_accuracy = leakage_metrics.get("accuracy", np.nan)
    print(f"Accuracy строгой модели: {strict_accuracy:.4f}.")
    print(f"Accuracy демонстрации утечки: {leakage_accuracy:.4f}.")
    print(
        "Интерпретация: если диагностические признаки резко улучшают качество, "
        "нужно проверить, не содержат ли они саму разметку или правило ее построения."
    )
"""),
        code(f"""
{depth_cell}
experiment_tree = DecisionTreeClassifier(max_depth=experiment_depth, min_samples_leaf=8, random_state=RANDOM_STATE)
experiment_tree.fit(X_train, y_train)
experiment_pred = experiment_tree.predict(X_test)
pd.Series(classification_metrics(y_test, experiment_pred), name="experiment")
"""),
        code("""
plt.figure(figsize=(16, 8))
plot_tree(
    tree_model,
    feature_names=classification_features,
    class_names=["недопустимый", "допустимый"],
    filled=True,
    rounded=True,
    impurity=True,
)
plt.title("Строгое дерево решений")
plt.show()
"""),
        md("""
## Типовые ошибки интерпретации

1. Оценивать дерево только по accuracy и не анализировать recall критического
   класса.
2. Использовать диагностическую метку отказа как обычный признак модели.
3. Увеличивать `max_depth` без анализа переобучения и объяснимости правил.
4. Записывать правила дерева без единиц измерения и физической интерпретации
   порога.

## Контрольные вопросы и мини-задания

1. Что означает класс 0 в данном блокноте и почему он критически важен?
2. Почему majority baseline может иметь приемлемую accuracy, но быть
   непригодным для диагностики?
3. Найдите глубину дерева с наилучшей F1-мерой критического класса и объясните
   компромисс между качеством и интерпретируемостью.
4. Измените `experiment_depth` и объясните, какие ошибки стали чаще или реже.
5. Выпишите одно правило дерева в инженерной форме: признак, порог, единица
   измерения и физический смысл.
"""),
        md("""
## Задание

1. Укажите целевую переменную, кодировку классов и физический смысл класса 0.
2. Сравните дерево решений с базовой моделью большинства.
3. Постройте матрицу ошибок и рассчитайте долю ложных разрешений.
4. Выпишите 2-3 правила дерева в инженерной форме.
5. Объясните, какие признаки были исключены из-за риска утечки данных.
"""),
    ]


def build_cells(config: DatasetConfig, lesson: int, teacher: bool) -> list[nbf.NotebookNode]:
    if lesson == 1:
        return lesson01_cells(config, teacher)
    if lesson == 2:
        return lesson02_cells(config, teacher)
    if lesson == 3:
        return lesson03_cells(config, teacher)
    raise ValueError(lesson)


def main() -> None:
    for config in DATASETS:
        for lesson in [1, 2, 3]:
            for teacher in [False, True]:
                suffix = "teacher" if teacher else "student"
                folder = TEACHER_DIR if teacher else STUDENT_DIR
                path = folder / f"{lesson:02d}_{config.short_name}_{suffix}.ipynb"
                write_notebook(path, build_cells(config, lesson, teacher))
                print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

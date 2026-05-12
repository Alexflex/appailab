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
    features_file: str
    diagnostics_file: str
    metadata_file: str
    group_column: str
    time_column: str
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
        features_file="external/zenodo_motor_temperature_features.csv",
        diagnostics_file="external/zenodo_motor_temperature_diagnostics.csv",
        metadata_file="external/zenodo_motor_temperature_metadata.md",
        group_column="profile_id",
        time_column="time_index",
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
            "target_temperature_c",
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
        features_file="external/zenodo_pmsm_inverter_fault_features.csv",
        diagnostics_file="external/zenodo_pmsm_inverter_fault_diagnostics.csv",
        metadata_file="external/zenodo_pmsm_inverter_fault_metadata.md",
        group_column="profile_id",
        time_column="time_index",
        lesson01_columns=[
            "speed_rpm",
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "ambient_temp_c",
            "temperature_c",
            "max_bridge_temp_c",
        ],
        regression_target="max_bridge_temp_c",
        regression_features=[
            "speed_rpm",
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "ambient_temp_c",
        ],
        regression_leakage_features=[
            "speed_rpm",
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "ambient_temp_c",
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
            "speed_rpm",
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "ambient_temp_c",
            "temperature_c",
        ],
        classification_leakage_features=[
            "speed_rpm",
            "torque_proxy_nm",
            "voltage_v",
            "current_a",
            "ambient_temp_c",
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
        features_file="external/mendeley_ev_powertrain_efficiency_features.csv",
        diagnostics_file="external/mendeley_ev_powertrain_efficiency_diagnostics.csv",
        metadata_file="external/mendeley_ev_powertrain_efficiency_metadata.md",
        group_column="profile_id",
        time_column="time_index",
        lesson01_columns=[
            "vehicle_speed_m_s",
            "acceleration_m_s2",
            "slope_rad",
            "motor_speed_rpm",
            "motor_torque_nm",
            "motor_efficiency",
            "drivetrain_efficiency",
        ],
        regression_target="drivetrain_efficiency",
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
            "motor_efficiency",
            "mechanical_power_w",
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
        regression_target_meaning="расчетный КПД электропривода транспортного средства",
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
DATASET_ID = "{config.dataset_id}"
DATASET_TITLE = "{config.title}"

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11


def load_external_or_fallback():
    if FEATURES_FILE.exists() and DIAGNOSTICS_FILE.exists():
        features = pd.read_csv(FEATURES_FILE)
        diagnostics = pd.read_csv(DIAGNOSTICS_FILE)
        source_status = "external"
    else:
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


def group_holdout_split(data, group_column, test_share=0.25):
    groups = np.array(sorted(data[group_column].dropna().unique()))
    test_count = max(1, int(np.ceil(len(groups) * test_share)))
    test_groups = groups[-test_count:]
    test_mask = data[group_column].isin(test_groups)
    return data.index[~test_mask], data.index[test_mask], test_groups


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
        else "# TODO: выберите столбцы для первичного анализа.\nanalysis_columns = []\nif not analysis_columns:\n    raise ValueError('Заполните analysis_columns.')"
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
        code(f"""
{todo}
features_df[analysis_columns].describe().T
"""),
        code(f"""
missing_table = (
    features_df.isna().sum().rename("missing_count").to_frame()
    .assign(missing_share=lambda x: x["missing_count"] / len(features_df))
)
missing_table[missing_table["missing_count"] > 0]
"""),
        code("""
numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
shown_columns = [column for column in analysis_columns if column in numeric_columns]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for ax, column in zip(axes, shown_columns[:6]):
    ax.hist(features_df[column].dropna(), bins=30, color="#4c78a8", edgecolor="white")
    ax.set_title(column)
    ax.set_ylabel("Число наблюдений")
for ax in axes[len(shown_columns[:6]):]:
    ax.set_axis_off()
plt.suptitle("Распределения основных величин", y=1.02)
plt.tight_layout()
plt.show()
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
plot_correlation_heatmap(
    features_df,
    [column for column in analysis_columns if column in features_df.columns],
    "Корреляции выбранных столбцов",
)
"""),
        code(f"""
x_column = analysis_columns[0]
y_column = "{config.regression_target}"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(features_df[x_column], features_df[y_column], alpha=0.55)
ax.set_xlabel(x_column)
ax.set_ylabel(y_column)
ax.set_title("Связь выбранного признака с целевой величиной")
plt.tight_layout()
plt.show()
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
        else "# TODO: заполните признаки строгой регрессионной модели.\nregression_features = []\nregression_target = '" + config.regression_target + "'\nif not regression_features:\n    raise ValueError('Заполните regression_features.')"
    )
    experiment_cell = (
        "experiment_degree = 2\nexperiment_alpha = 1.0"
        if teacher
        else "experiment_degree = None\nexperiment_alpha = None\nif experiment_degree is None or experiment_alpha is None:\n    raise ValueError('Задайте experiment_degree и experiment_alpha.')"
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
"""),
code(setup_cell(config)),
        code(f"""
{feature_cell}
forbidden_regression_features = set({forbidden!r})
leaked_regression_features = forbidden_regression_features & set(regression_features)
if leaked_regression_features:
    raise AssertionError(
        "Обнаружена утечка данных в regression_features: "
        f"{{sorted(leaked_regression_features)}}"
    )

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
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
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
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
        code(f"""
{experiment_cell}
experiment_model = Pipeline([
    ("poly", PolynomialFeatures(degree=experiment_degree, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=experiment_alpha)),
])
experiment_model.fit(X_train, y_train)
experiment_pred = experiment_model.predict(X_test)
pd.Series(regression_metrics(y_test, experiment_pred), name="experiment")
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
        else "# TODO: заполните признаки строгой классификационной модели.\nclassification_features = []\nclassification_target = '" + config.classification_target + "'\nif not classification_features:\n    raise ValueError('Заполните classification_features.')"
    )
    depth_cell = (
        "experiment_depth = 3"
        if teacher
        else "experiment_depth = None\nif experiment_depth is None:\n    raise ValueError('Задайте experiment_depth.')"
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
"""),
code(setup_cell(config)),
        code(f"""
{feature_cell}
forbidden_classification_features = set({forbidden!r})
leaked_classification_features = forbidden_classification_features & set(classification_features)
if leaked_classification_features:
    raise AssertionError(
        "Обнаружена утечка данных в classification_features: "
        f"{{sorted(leaked_classification_features)}}"
    )

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

model_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=classification_features + [classification_target]
).copy()
train_idx, test_idx, test_groups = group_holdout_split(model_df, GROUP_COLUMN, test_share=0.25)
X_train = model_df.loc[train_idx, classification_features]
X_test = model_df.loc[test_idx, classification_features]
y_train = model_df.loc[train_idx, classification_target].astype(int)
y_test = model_df.loc[test_idx, classification_target].astype(int)

print("Тестовые группы:", test_groups)
print("Распределение классов в полной выборке:")
print(model_df[classification_target].value_counts().sort_index())
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
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["недопустимый", "допустимый"])
disp.plot(values_format="d", cmap="Blues")
plt.title("Матрица ошибок строгого дерева решений")
plt.show()
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
    train_idx_l, test_idx_l, _ = group_holdout_split(leakage_df, GROUP_COLUMN, test_share=0.25)
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

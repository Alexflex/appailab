"""Генераторы учебных инженерных наборов данных.

Модуль используется в практических занятиях 1-3. Он формирует небольшие
таблицы, которые можно обрабатывать в аудитории за ограниченное время.

Методическое допущение:
данные являются учебными и синтетическими, но структура признаков согласована
с открытым набором Electric Motor Temperature, где представлены измерения
постоянно-магнитной синхронной машины (Permanent Magnet Synchronous Motor,
PMSM) на лабораторном стенде. Поэтому набор пригоден для обучения методам
анализа данных, но не является заменой сертификационных испытаний машины.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 20260507


@dataclass(frozen=True)
class DatasetPaths:
    """Пути к сформированным учебным наборам данных."""

    practice_01: Path
    practice_02: Path
    practice_02_features: Path
    practice_02_diagnostics: Path
    practice_03: Path
    practice_03_features: Path
    practice_03_diagnostics: Path
    catalog: Path
    assignments: Path
    metadata: Path


def _torque_limit_from_speed(speed_rpm: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Рассчитать учебный предел момента по скорости вращения."""

    return np.clip(0.115 - 1.1e-6 * (speed_rpm - 5_000), 0.035, 0.120)


def _winding_resistance_from_temperature(
    temperature_c: pd.Series | np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.006,
) -> pd.Series | np.ndarray:
    """Рассчитать сопротивление обмотки с малым шумом измерения."""

    nominal = 0.18 * (1.0 + 0.0039 * (temperature_c - 20.0))
    return nominal * rng.normal(1.0, noise_std, len(nominal))


def _refresh_power_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Пересчитать мощность, потери и КПД после изменения режима."""

    result = df.copy()
    omega_rad_s = 2.0 * np.pi * result["speed_rpm"] / 60.0
    result["output_power_w"] = result["torque_nm"] * omega_rad_s
    input_power_w = result["voltage_v"] * result["current_a"]
    result["loss_power_w"] = input_power_w - result["output_power_w"]
    result["efficiency"] = result["output_power_w"] / input_power_w
    return result


def generate_motor_measurements(
    n_samples: int = 240,
    random_state: int = RANDOM_SEED,
    include_quality_artifacts: bool = True,
    allow_torque_violation: bool = False,
) -> pd.DataFrame:
    """Сформировать таблицу режимов маломощного высокооборотного двигателя.

    Parameters
    ----------
    n_samples:
        Число наблюдений. Наблюдение - одна строка таблицы, соответствующая
        одному расчетному или измеренному режиму.
    random_state:
        Начальное состояние генератора случайных чисел. Фиксация этого
        параметра обеспечивает воспроизводимость результатов.
    include_quality_artifacts:
        Если значение равно True, в таблицу добавляются пропуски и выбросы.
        Они нужны для занятия 1, где изучаются первичная проверка данных и
        простейшая очистка.
    allow_torque_violation:
        Если значение равно True, часть режимов может немного превышать
        учебный предел момента. Это используется только в задаче
        классификации допустимых и недопустимых режимов.
    """

    rng = np.random.default_rng(random_state)

    profile_count = 6
    profile_id = np.repeat(np.arange(1, profile_count + 1), np.ceil(n_samples / profile_count))[:n_samples]
    rng.shuffle(profile_id)

    # Профили задают условные группы режимов. Они не являются временным рядом,
    # но позволяют показать, что разные испытательные условия могут иметь
    # разные распределения скорости, нагрузки и температуры.
    speed_profile_centers = np.array([9_000.0, 18_000.0, 30_000.0, 42_000.0, 54_000.0, 62_000.0])
    load_profile_centers = np.array([0.28, 0.42, 0.62, 0.78, 0.58, 0.38])

    speed_rpm = rng.normal(speed_profile_centers[profile_id - 1], 4_200.0, n_samples)
    speed_rpm = np.clip(speed_rpm, 5_000.0, 65_000.0)

    # Максимально допустимый момент снижается на высоких скоростях. Это
    # упрощенное представление ограничения по мощности и тепловому режиму.
    torque_limit_nm = _torque_limit_from_speed(speed_rpm)
    load_upper = 1.02 if allow_torque_violation else 1.00
    load_fraction = rng.normal(load_profile_centers[profile_id - 1], 0.13, n_samples)
    load_fraction = np.clip(load_fraction, 0.10, load_upper)
    torque_nm = torque_limit_nm * load_fraction

    # Дискретные уровни напряжения близки к типовым низковольтным приводам.
    voltage_nominal = rng.choice([24.0, 36.0, 48.0, 60.0], size=n_samples, p=[0.20, 0.30, 0.35, 0.15])
    voltage_v = voltage_nominal + rng.normal(0.0, 0.45, n_samples)

    omega_rad_s = 2.0 * np.pi * speed_rpm / 60.0
    output_power_w = torque_nm * omega_rad_s

    # КПД задается как гладкая функция скорости и относительной нагрузки.
    # Пик КПД находится в средней области скоростей и нагрузок, а на краях
    # карты режимов эффективность снижается.
    speed_efficiency_factor = np.clip(1.0 - ((speed_rpm - 36_000.0) / 38_000.0) ** 2, 0.0, 1.0)
    load_efficiency_factor = np.clip(1.0 - ((load_fraction - 0.68) / 0.72) ** 2, 0.0, 1.0)
    efficiency_clean = 0.42 + 0.46 * speed_efficiency_factor * load_efficiency_factor
    efficiency_clean += rng.normal(0.0, 0.018, n_samples)
    efficiency_clean = np.clip(efficiency_clean, 0.25, 0.92)

    ideal_input_power_w = output_power_w / efficiency_clean
    ideal_loss_power_w = ideal_input_power_w - output_power_w

    current_a = ideal_input_power_w / voltage_v + rng.normal(0.0, 0.16, n_samples)
    minimum_current_a = output_power_w / (0.96 * voltage_v)
    current_a = np.maximum(current_a, minimum_current_a)
    current_a = np.clip(current_a, 0.15, None)

    measured_input_power_w = voltage_v * current_a
    loss_power_w = measured_input_power_w - output_power_w

    ambient_temp_c = rng.uniform(20.0, 35.0, n_samples)
    temperature_c = ambient_temp_c + 0.43 * ideal_loss_power_w + 0.025 * current_a**2
    temperature_c += rng.normal(0.0, 2.2, n_samples)

    winding_resistance_ohm = _winding_resistance_from_temperature(temperature_c, rng)
    measured_efficiency = output_power_w / measured_input_power_w

    df = pd.DataFrame(
        {
            "sample_id": np.arange(1, n_samples + 1),
            "profile_id": profile_id,
            "speed_rpm": speed_rpm,
            "torque_nm": torque_nm,
            "voltage_v": voltage_v,
            "current_a": current_a,
            "ambient_temp_c": ambient_temp_c,
            "winding_resistance_ohm": winding_resistance_ohm,
            "temperature_c": temperature_c,
            "output_power_w": output_power_w,
            "loss_power_w": loss_power_w,
            "efficiency": measured_efficiency,
            "torque_limit_nm": torque_limit_nm,
        }
    )

    if include_quality_artifacts:
        df = _add_missing_values_and_outliers(df, rng)

    return _round_engineering_columns(df)


def _add_missing_values_and_outliers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Добавить контролируемые дефекты качества данных.

    В реальной лабораторной таблице пропуски могут возникать из-за сбоя
    датчика или обмена данными, а выбросы - из-за кратковременного переходного
    процесса, ошибки синхронизации или некорректной регистрации. В учебном
    наборе такие случаи добавляются явно, чтобы студент мог обнаружить их
    методами первичного анализа.
    """

    result = df.copy()

    nullable_columns = ["torque_nm", "current_a", "temperature_c", "efficiency"]
    for column in nullable_columns:
        missing_count = max(1, int(0.025 * len(result)))
        rows = rng.choice(result.index, size=missing_count, replace=False)
        result.loc[rows, column] = np.nan

    outlier_rows = rng.choice(result.index, size=max(3, int(0.025 * len(result))), replace=False)
    half = len(outlier_rows) // 2
    current_outlier_rows = outlier_rows[:half]
    result.loc[current_outlier_rows, "current_a"] *= rng.uniform(1.65, 2.20, size=half)
    valid_current_rows = result.loc[current_outlier_rows].dropna(
        subset=["voltage_v", "current_a", "output_power_w"]
    ).index
    input_power_w = result.loc[valid_current_rows, "voltage_v"] * result.loc[valid_current_rows, "current_a"]
    result.loc[valid_current_rows, "loss_power_w"] = input_power_w - result.loc[valid_current_rows, "output_power_w"]
    result.loc[valid_current_rows, "efficiency"] = (
        result.loc[valid_current_rows, "output_power_w"] / input_power_w
    )
    result.loc[outlier_rows[half:], "temperature_c"] += rng.uniform(25.0, 42.0, size=len(outlier_rows) - half)

    return result


def generate_drive_mode_classification(
    n_samples: int = 420,
    random_state: int = RANDOM_SEED + 3,
) -> pd.DataFrame:
    """Сформировать набор данных для классификации допустимости режима.

    Целевая переменная `is_allowed` задает бинарную классификацию:
    1 - режим допустим, 0 - режим требует ограничения или отключения.

    Дополнительный столбец `mode_label` хранит инженерную причину ограничения.
    Он используется для интерпретации, но не должен включаться в признаки
    модели, поскольку содержит информацию, близкую к целевой переменной.
    """

    if n_samples < 125:
        raise ValueError("Для сбалансированной классификации требуется не менее 125 наблюдений.")

    rng = np.random.default_rng(random_state)
    cause_count = max(25, int(round(n_samples * 0.06)))
    cause_count = min(cause_count, n_samples // 6)
    allowed_count = n_samples - 5 * cause_count

    df = generate_motor_measurements(
        n_samples=n_samples,
        random_state=random_state,
        include_quality_artifacts=False,
        allow_torque_violation=False,
    )

    # Пороговые значения выбраны как учебные эксплуатационные ограничения.
    # Генератор ниже намеренно формирует примерно одинаковое число примеров
    # каждой причины нарушения, чтобы дерево решений не вырождалось в одно
    # правило по скорости или КПД.
    current_limit_a = 11.0
    temperature_limit_c = 74.0
    speed_limit_rpm = 63_000.0
    min_efficiency = 0.45

    result = df.copy()

    # Сначала все строки приводятся к безопасной области. Затем отдельные
    # группы строк переводятся в контролируемые нарушения одного типа.
    result["speed_rpm"] = np.clip(result["speed_rpm"], 8_000.0, speed_limit_rpm - 5_000.0)
    result["torque_limit_nm"] = _torque_limit_from_speed(result["speed_rpm"])
    safe_load_fraction = rng.uniform(0.35, 0.72, n_samples)
    result["torque_nm"] = result["torque_limit_nm"] * safe_load_fraction
    safe_efficiency = rng.uniform(0.62, 0.80, n_samples)
    omega_rad_s = 2.0 * np.pi * result["speed_rpm"] / 60.0
    result["output_power_w"] = result["torque_nm"] * omega_rad_s
    result["current_a"] = result["output_power_w"] / (result["voltage_v"] * safe_efficiency)
    high_current_mask = result["current_a"] > (current_limit_a - 1.0)
    if high_current_mask.any():
        scale = (current_limit_a - 1.0) / result.loc[high_current_mask, "current_a"]
        result.loc[high_current_mask, "torque_nm"] *= scale * 0.95
        omega_rad_s = 2.0 * np.pi * result.loc[high_current_mask, "speed_rpm"] / 60.0
        output_power_w = result.loc[high_current_mask, "torque_nm"] * omega_rad_s
        result.loc[high_current_mask, "current_a"] = (
            output_power_w / (result.loc[high_current_mask, "voltage_v"] * safe_efficiency[high_current_mask])
        )
    result = _refresh_power_columns(result)
    result["temperature_c"] = np.minimum(
        result["ambient_temp_c"] + rng.uniform(10.0, 32.0, n_samples),
        temperature_limit_c - 3.0,
    )
    result["winding_resistance_ohm"] = _winding_resistance_from_temperature(result["temperature_c"], rng)

    indexes = result.index.to_numpy()
    overcurrent_idx = indexes[allowed_count : allowed_count + cause_count]
    overheating_idx = indexes[allowed_count + cause_count : allowed_count + 2 * cause_count]
    overspeed_idx = indexes[allowed_count + 2 * cause_count : allowed_count + 3 * cause_count]
    low_efficiency_idx = indexes[allowed_count + 3 * cause_count : allowed_count + 4 * cause_count]
    torque_violation_idx = indexes[allowed_count + 4 * cause_count : allowed_count + 5 * cause_count]

    # Превышение тока без одновременного срабатывания остальных причин.
    size = len(overcurrent_idx)
    result.loc[overcurrent_idx, "speed_rpm"] = rng.uniform(50_000.0, 53_000.0, size)
    result.loc[overcurrent_idx, "voltage_v"] = rng.uniform(53.0, 56.0, size)
    result.loc[overcurrent_idx, "current_a"] = rng.uniform(current_limit_a + 0.25, current_limit_a + 0.55, size)
    result.loc[overcurrent_idx, "torque_limit_nm"] = _torque_limit_from_speed(result.loc[overcurrent_idx, "speed_rpm"])
    target_efficiency = rng.uniform(min_efficiency + 0.070, min_efficiency + 0.105, size)
    target_output_power = result.loc[overcurrent_idx, "voltage_v"] * result.loc[overcurrent_idx, "current_a"] * target_efficiency
    omega_rad_s = 2.0 * np.pi * result.loc[overcurrent_idx, "speed_rpm"] / 60.0
    result.loc[overcurrent_idx, "torque_nm"] = target_output_power / omega_rad_s
    result.loc[overcurrent_idx, "torque_nm"] = np.minimum(
        result.loc[overcurrent_idx, "torque_nm"],
        result.loc[overcurrent_idx, "torque_limit_nm"] * 0.92,
    )
    result.loc[overcurrent_idx, "temperature_c"] = result.loc[overcurrent_idx, "ambient_temp_c"] + rng.uniform(18.0, 30.0, size)

    # Перегрев при безопасных токе, скорости и КПД.
    size = len(overheating_idx)
    result.loc[overheating_idx, "temperature_c"] = rng.uniform(temperature_limit_c + 2.0, temperature_limit_c + 8.0, size)
    result.loc[overheating_idx, "current_a"] = np.minimum(result.loc[overheating_idx, "current_a"], current_limit_a - 1.2)

    # Превышение скорости при малом моменте.
    size = len(overspeed_idx)
    result.loc[overspeed_idx, "speed_rpm"] = rng.uniform(speed_limit_rpm + 700.0, speed_limit_rpm + 2_000.0, size)
    result.loc[overspeed_idx, "voltage_v"] = rng.uniform(58.0, 66.0, size)
    result.loc[overspeed_idx, "torque_limit_nm"] = _torque_limit_from_speed(result.loc[overspeed_idx, "speed_rpm"])
    result.loc[overspeed_idx, "torque_nm"] = rng.uniform(0.018, 0.026, size)
    target_efficiency = rng.uniform(0.64, 0.76, size)
    omega_rad_s = 2.0 * np.pi * result.loc[overspeed_idx, "speed_rpm"] / 60.0
    target_output_power = result.loc[overspeed_idx, "torque_nm"] * omega_rad_s
    result.loc[overspeed_idx, "current_a"] = target_output_power / (result.loc[overspeed_idx, "voltage_v"] * target_efficiency)
    result.loc[overspeed_idx, "temperature_c"] = result.loc[overspeed_idx, "ambient_temp_c"] + rng.uniform(14.0, 28.0, size)

    # Низкий КПД при безопасных первичных ограничениях.
    size = len(low_efficiency_idx)
    result.loc[low_efficiency_idx, "speed_rpm"] = rng.uniform(22_000.0, 42_000.0, size)
    result.loc[low_efficiency_idx, "torque_limit_nm"] = _torque_limit_from_speed(result.loc[low_efficiency_idx, "speed_rpm"])
    result.loc[low_efficiency_idx, "torque_nm"] = result.loc[low_efficiency_idx, "torque_limit_nm"] * rng.uniform(0.28, 0.48, size)
    target_efficiency = rng.uniform(0.32, min_efficiency - 0.035, size)
    omega_rad_s = 2.0 * np.pi * result.loc[low_efficiency_idx, "speed_rpm"] / 60.0
    target_output_power = result.loc[low_efficiency_idx, "torque_nm"] * omega_rad_s
    result.loc[low_efficiency_idx, "current_a"] = target_output_power / (
        result.loc[low_efficiency_idx, "voltage_v"] * target_efficiency
    )
    high_current_mask = result.loc[low_efficiency_idx, "current_a"] > (current_limit_a - 1.1)
    if high_current_mask.any():
        high_indexes = high_current_mask[high_current_mask].index
        scale = (current_limit_a - 1.1) / result.loc[high_indexes, "current_a"]
        result.loc[high_indexes, "torque_nm"] *= scale * 0.90
        omega_rad_s = 2.0 * np.pi * result.loc[high_indexes, "speed_rpm"] / 60.0
        target_output_power = result.loc[high_indexes, "torque_nm"] * omega_rad_s
        result.loc[high_indexes, "current_a"] = target_output_power / (
            result.loc[high_indexes, "voltage_v"] * target_efficiency[high_current_mask.to_numpy()]
        )
    result.loc[low_efficiency_idx, "temperature_c"] = result.loc[low_efficiency_idx, "ambient_temp_c"] + rng.uniform(12.0, 25.0, size)

    # Превышение ограничения момента при безопасных токе, скорости и КПД.
    size = len(torque_violation_idx)
    result.loc[torque_violation_idx, "speed_rpm"] = rng.uniform(18_000.0, 32_000.0, size)
    result.loc[torque_violation_idx, "voltage_v"] = rng.uniform(58.0, 66.0, size)
    result.loc[torque_violation_idx, "torque_limit_nm"] = _torque_limit_from_speed(result.loc[torque_violation_idx, "speed_rpm"])
    result.loc[torque_violation_idx, "torque_nm"] = result.loc[torque_violation_idx, "torque_limit_nm"] * rng.uniform(1.05, 1.12, size)
    target_efficiency = rng.uniform(0.62, 0.74, size)
    omega_rad_s = 2.0 * np.pi * result.loc[torque_violation_idx, "speed_rpm"] / 60.0
    target_output_power = result.loc[torque_violation_idx, "torque_nm"] * omega_rad_s
    result.loc[torque_violation_idx, "current_a"] = target_output_power / (
        result.loc[torque_violation_idx, "voltage_v"] * target_efficiency
    )
    result.loc[torque_violation_idx, "temperature_c"] = result.loc[torque_violation_idx, "ambient_temp_c"] + rng.uniform(16.0, 30.0, size)

    result = _refresh_power_columns(result)
    result["winding_resistance_ohm"] = _winding_resistance_from_temperature(result["temperature_c"], rng)

    overcurrent = result["current_a"] > current_limit_a
    overheating = result["temperature_c"] > temperature_limit_c
    overspeed = result["speed_rpm"] > speed_limit_rpm
    low_efficiency = result["efficiency"] < min_efficiency
    torque_violation = result["torque_nm"] > result["torque_limit_nm"]

    is_allowed = ~(overcurrent | overheating | overspeed | low_efficiency | torque_violation)

    conditions = [
        overheating,
        overcurrent,
        overspeed,
        low_efficiency,
        torque_violation,
    ]
    labels = [
        "thermal_limit",
        "current_limit",
        "speed_limit",
        "low_efficiency",
        "torque_limit",
    ]
    mode_label = np.select(conditions, labels, default="allowed")

    result["overcurrent"] = overcurrent.astype(int)
    result["overheating"] = overheating.astype(int)
    result["overspeed"] = overspeed.astype(int)
    result["low_efficiency"] = low_efficiency.astype(int)
    result["torque_violation"] = torque_violation.astype(int)
    result["violation_count"] = result[
        ["overcurrent", "overheating", "overspeed", "low_efficiency", "torque_violation"]
    ].sum(axis=1)
    result["current_margin_a"] = current_limit_a - result["current_a"]
    result["temperature_margin_c"] = temperature_limit_c - result["temperature_c"]
    result["speed_margin_rpm"] = speed_limit_rpm - result["speed_rpm"]
    result["is_allowed"] = is_allowed.astype(int)
    result["mode_label"] = mode_label

    return _round_engineering_columns(result)


def _round_engineering_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Округлить численные столбцы до точности, удобной для учебных таблиц."""

    result = df.copy()
    decimals = {
        "speed_rpm": 1,
        "torque_nm": 5,
        "voltage_v": 3,
        "current_a": 4,
        "ambient_temp_c": 3,
        "winding_resistance_ohm": 6,
        "temperature_c": 3,
        "output_power_w": 3,
        "loss_power_w": 3,
        "efficiency": 5,
        "torque_limit_nm": 5,
        "current_margin_a": 4,
        "temperature_margin_c": 3,
        "speed_margin_rpm": 1,
    }
    for column, digits in decimals.items():
        if column in result.columns:
            result[column] = result[column].round(digits)
    return result


def _practice_02_feature_columns() -> list[str]:
    """Столбцы безопасной учебной таблицы для регрессии КПД."""

    return [
        "sample_id",
        "profile_id",
        "speed_rpm",
        "torque_nm",
        "voltage_v",
        "temperature_c",
        "ambient_temp_c",
        "efficiency",
    ]


def _practice_02_diagnostic_columns() -> list[str]:
    """Столбцы, полезные для диагностики, но опасные как признаки модели."""

    return [
        "sample_id",
        "current_a",
        "winding_resistance_ohm",
        "output_power_w",
        "loss_power_w",
        "torque_limit_nm",
    ]


def _practice_03_feature_columns() -> list[str]:
    """Столбцы безопасной учебной таблицы для классификации режимов."""

    return [
        "sample_id",
        "profile_id",
        "speed_rpm",
        "torque_nm",
        "voltage_v",
        "current_a",
        "ambient_temp_c",
        "temperature_c",
        "is_allowed",
    ]


def _practice_03_diagnostic_columns() -> list[str]:
    """Диагностические столбцы правила разметки занятия 3."""

    return [
        "sample_id",
        "winding_resistance_ohm",
        "output_power_w",
        "loss_power_w",
        "efficiency",
        "torque_limit_nm",
        "overcurrent",
        "overheating",
        "overspeed",
        "low_efficiency",
        "torque_violation",
        "violation_count",
        "current_margin_a",
        "temperature_margin_c",
        "speed_margin_rpm",
        "mode_label",
    ]


def create_all_datasets(output_dir: str | Path) -> DatasetPaths:
    """Создать CSV-файлы для первых трех практических занятий."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    practice_01 = output_path / "practice_01_motor_measurements.csv"
    practice_02 = output_path / "practice_02_motor_efficiency.csv"
    practice_02_features = output_path / "practice_02_motor_efficiency_features.csv"
    practice_02_diagnostics = output_path / "practice_02_motor_efficiency_diagnostics.csv"
    practice_03 = output_path / "practice_03_drive_mode_classification.csv"
    practice_03_features = output_path / "practice_03_drive_mode_features.csv"
    practice_03_diagnostics = output_path / "practice_03_drive_mode_diagnostics.csv"
    catalog = output_path / "practice_01_03_dataset_catalog.csv"
    assignments = output_path / "practice_01_03_dataset_assignments.csv"
    metadata = output_path / "DATASETS.md"

    generate_motor_measurements(
        n_samples=240,
        random_state=RANDOM_SEED,
        include_quality_artifacts=True,
    ).to_csv(practice_01, index=False)

    practice_02_df = generate_motor_measurements(
        n_samples=300,
        random_state=RANDOM_SEED + 2,
        include_quality_artifacts=False,
    )
    practice_02_df.to_csv(practice_02, index=False)
    practice_02_df[_practice_02_feature_columns()].to_csv(practice_02_features, index=False)
    practice_02_df[_practice_02_diagnostic_columns()].to_csv(practice_02_diagnostics, index=False)

    practice_03_df = generate_drive_mode_classification(
        n_samples=420,
        random_state=RANDOM_SEED + 3,
    )
    practice_03_df.to_csv(practice_03, index=False)
    practice_03_df[_practice_03_feature_columns()].to_csv(practice_03_features, index=False)
    practice_03_df[_practice_03_diagnostic_columns()].to_csv(practice_03_diagnostics, index=False)

    dataset_catalog().to_csv(catalog, index=False)
    dataset_assignments().to_csv(assignments, index=False)
    metadata.write_text(_dataset_metadata_text(), encoding="utf-8")

    return DatasetPaths(
        practice_01=practice_01,
        practice_02=practice_02,
        practice_02_features=practice_02_features,
        practice_02_diagnostics=practice_02_diagnostics,
        practice_03=practice_03,
        practice_03_features=practice_03_features,
        practice_03_diagnostics=practice_03_diagnostics,
        catalog=catalog,
        assignments=assignments,
        metadata=metadata,
    )


def dataset_catalog() -> pd.DataFrame:
    """Вернуть реестр найденных наборов данных для занятий 1-3.

    Реестр используется базовыми блокнотами как обязательная часть работы:
    студенты анализируют не только учебный CSV, но и карту реальных источников,
    выбирают подходящий набор для расширения и фиксируют ограничения доступа.
    Сырые данные из крупных архивов не загружаются автоматически.
    """

    rows = [
        {
            "dataset_id": "zenodo_motor_temperature",
            "name": "ElectricMotorTemperature",
            "url": "https://zenodo.org/records/11235562",
            "object": "электродвигатель, представленный многомерными временными рядами (multivariate time series)",
            "license": "CC BY 4.0",
            "access": "прямая загрузка Zenodo",
            "size_note": "TRAIN около 104 МБ, TEST около 45 МБ; фрагменты длиной 60 отсчетов",
            "format_note": "TSML .ts",
            "lessons": "1,2,3",
            "base_usage": "реализованный внешний блокнот; используется для анализа временных признаков и регрессии температуры",
            "risk_level": "средний",
            "risk_note": "физические имена каналов в архиве не заданы; требуется групповое разбиение по profile_id",
        },
        {
            "dataset_id": "pmsm_inverter_fault_zenodo",
            "name": "Comprehensive Dataset for Fault Detection and Diagnosis in Inverter-Driven PMSM Systems",
            "url": "https://zenodo.org/records/14482932",
            "object": "PMSM-инвертор, лабораторная диагностика",
            "license": "CC BY 4.0",
            "access": "прямая загрузка Zenodo",
            "size_note": "компактный архив около 1.1 МБ по данным карточки источника",
            "format_note": "CSV / zip",
            "lessons": "1,2,3",
            "base_usage": "расширение для диагностики и регрессии температурных признаков",
            "risk_level": "низкий",
            "risk_note": "исходная постановка ближе к диагностике, чем к расчету КПД",
        },
        {
            "dataset_id": "ev_powertrain_efficiency",
            "name": "Processed Data for EV Powertrain Efficiency",
            "url": "https://data.mendeley.com/datasets/kbwr2z8r3y",
            "object": "электропривод транспортного средства",
            "license": "CC BY 4.0",
            "access": "Mendeley Data",
            "size_note": "размер на карточке не указан",
            "format_note": "CSV и Jupyter Notebook",
            "lessons": "1,2",
            "base_usage": "дополнительный источник по энергетической эффективности",
            "risk_level": "средний",
            "risk_note": "КПД расчетный, а не прямое стендовое измерение двигателя",
        },
        {
            "dataset_id": "system_identification_electric_motor",
            "name": "Identifying the Physics Behind an Electric Motor",
            "url": "https://www.kaggle.com/datasets/hankelea/system-identification-of-an-electric-motor",
            "object": "PMSM и двухуровневый IGBT-инвертор",
            "license": "Data files copyright original authors; условия переиздания неясны",
            "access": "Kaggle",
            "size_note": "около 40 млн образцов",
            "format_note": "табличные файлы Kaggle",
            "lessons": "1,2",
            "base_usage": "продвинутый источник для системной идентификации",
            "risk_level": "высокий",
            "risk_note": "очень большой объем; неясная лицензия; нет температур и КПД",
        },
        {
            "dataset_id": "ipmsm_motor_parameters",
            "name": "Dataset for motor parameters of IPMSM",
            "url": "https://www.kaggle.com/datasets/uuuuuuuuu/dataset-motor-parameters-ipmsm/data",
            "object": "параметры и геометрия синхронного двигателя с внутренним расположением постоянных магнитов (Interior Permanent Magnet Synchronous Motor, IPMSM)",
            "license": "CC BY-SA 4.0",
            "access": "Kaggle",
            "size_note": "CSV и изображения геометрии",
            "format_note": "CSV, PNG",
            "lessons": "1,2",
            "base_usage": "дополнительный источник по суррогатным моделям (surrogate models) параметров двигателя",
            "risk_level": "средний",
            "risk_note": "данные расчетно-синтетические, не стендовые",
        },
        {
            "dataset_id": "pmsm_foc_mendeley",
            "name": "PMSM_FOC dataset",
            "url": "https://data.mendeley.com/datasets/6tjkgtfnky/2",
            "object": "PMSM с Field Oriented Control",
            "license": "CC BY 4.0",
            "access": "Mendeley Data",
            "size_note": "размер на карточке не указан",
            "format_note": "MAT, JPG",
            "lessons": "1,2,3",
            "base_usage": "модельная динамика управления; опциональная регрессия ошибки скорости",
            "risk_level": "средний",
            "risk_note": "симуляционные данные; требуется scipy.io.loadmat",
        },
        {
            "dataset_id": "permanent_magnet_dc_worm_motor",
            "name": "Permanent Magnet DC Worm Geared Motor Data",
            "url": "https://data.mendeley.com/datasets/2rkpsss6fd/2",
            "object": "двигатель постоянного тока с червячным редуктором",
            "license": "CC BY 4.0",
            "access": "Mendeley Data",
            "size_note": "около 27 МБ по статье Data in Brief",
            "format_note": "Excel / raw files",
            "lessons": "1,2,3",
            "base_usage": "понятные признаки: напряжение, ток, скорость; хорош для вводного анализа",
            "risk_level": "средний",
            "risk_note": "нужна предварительная нормализация листов Excel",
        },
        {
            "dataset_id": "uci_ai4i_2020",
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "url": "https://archive.ics.uci.edu/dataset/601/ai4i",
            "object": "синтетическая промышленная установка",
            "license": "CC BY 4.0",
            "access": "прямой UCI-доступ без регистрации",
            "size_note": "10 000 строк, 14 столбцов, около 510 КБ",
            "format_note": "CSV / UCI API",
            "lessons": "1,3",
            "base_usage": "малый дидактический набор для дерева решений",
            "risk_level": "низкий",
            "risk_note": "синтетическая природа; объект не является электродвигателем",
        },
        {
            "dataset_id": "figshare_three_phase_induction_fault",
            "name": "MOTOR FAULT DETECTION DATA",
            "url": "https://figshare.com/articles/dataset/MOTOR_FAULT_DETECTION_DATA/27216219",
            "object": "трехфазный асинхронный двигатель",
            "license": "CC BY 4.0",
            "access": "прямая загрузка Figshare",
            "size_note": "около 5.16 ГБ, 10 CSV-файлов",
            "format_note": "CSV",
            "lessons": "1,3",
            "base_usage": "расширенная классификация отказов после оконной агрегации",
            "risk_level": "высокий",
            "risk_note": "большой объем; сырые сигналы требуют извлечения признаков",
        },
        {
            "dataset_id": "paderborn_bearing",
            "name": "Paderborn University Bearing Data Center",
            "url": "https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter",
            "object": "электромеханический привод с подшипниковыми дефектами",
            "license": "CC BY-NC 4.0",
            "access": "прямые MATLAB-файлы с сайта университета",
            "size_note": "32 состояния, 4 режима, 20 измерений на режим",
            "format_note": "MAT",
            "lessons": "1,3",
            "base_usage": "академический источник для диагностики по токам и вибрациям",
            "risk_level": "высокий",
            "risk_note": "некоммерческая лицензия; MATLAB-формат; риск утечки по фрагментам одного подшипника",
        },
        {
            "dataset_id": "mendeley_rotating_machine_fault",
            "name": "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine",
            "url": "https://data.mendeley.com/datasets/ztmf3m7h5x/6",
            "object": "вращающаяся машина при дефектах и разных нагрузках",
            "license": "CC BY 4.0",
            "access": "Mendeley Data",
            "size_note": "версия 6, размер зависит от выбранных архивов",
            "format_note": "MAT, TDMS",
            "lessons": "1,3",
            "base_usage": "мультимодальная диагностика после извлечения признаков",
            "risk_level": "высокий",
            "risk_note": "требуются чтение MAT/TDMS и предварительная агрегация сигналов",
        },
        {
            "dataset_id": "cwru_bearing",
            "name": "Case Western Reserve University Bearing Data Center",
            "url": "https://engineering.case.edu/bearingdatacenter/welcome",
            "object": "подшипники двигателя Reliance Electric мощностью 2 л. с. (около 1,49 кВт)",
            "license": "явная лицензия на официальной странице не указана",
            "access": "публичные MATLAB-файлы",
            "size_note": "данные 12/48 кГц; размер зависит от выбранных файлов",
            "format_note": "MAT",
            "lessons": "1,3",
            "base_usage": "классический эталонный набор для диагностики подшипников",
            "risk_level": "средний",
            "risk_note": "искусственные дефекты электроэрозионной обработки; неясные условия переиспользования",
        },
        {
            "dataset_id": "university_ottawa_uoemd",
            "name": "University of Ottawa Electric Motor Dataset UOEMD-VAFCVS",
            "url": "https://data.mendeley.com/datasets/msxs4vj48g/1",
            "object": "электродвигатели с вибрацией, акустикой и температурой",
            "license": "CC BY 4.0",
            "access": "Mendeley Data",
            "size_note": "128 наборов, 10 секунд, 42 кГц",
            "format_note": "сигнальные файлы",
            "lessons": "1,3",
            "base_usage": "классификация типов неисправностей после извлечения признаков",
            "risk_level": "высокий",
            "risk_note": "искусственные дефекты; требуется обработка сигналов",
        },
        {
            "dataset_id": "pmsm_stator_faults",
            "name": "Vibration and Current Dataset of Three-Phase PMSMs with Stator Faults",
            "url": "https://data.mendeley.com/datasets/rgn5brrgrn/5",
            "object": "PMSM с межвитковыми и межкатушечными КЗ",
            "license": "CC BY 4.0",
            "access": "Mendeley Data",
            "size_note": "высокочастотные TDMS-файлы",
            "format_note": "TDMS",
            "lessons": "1,3",
            "base_usage": "продвинутый источник для диагностики дефектов статора",
            "risk_level": "высокий",
            "risk_note": "требуется библиотека для TDMS и расчет сигналовых признаков",
        },
        {
            "dataset_id": "mcc5_thu_motor",
            "name": "MCC5-THU Multi-mode Fault Diagnosis",
            "url": "https://data.mendeley.com/datasets/6s3dggj9mw/1",
            "object": "трехфазный асинхронный двигатель с многими типами отказов",
            "license": "CC BY 4.0",
            "access": "Mendeley Data",
            "size_note": "282 запуска по 90 с, около 13 ГБ по зеркалу",
            "format_note": "сигнальные файлы",
            "lessons": "3",
            "base_usage": "исследовательский проект по многоклассовой диагностике",
            "risk_level": "высокий",
            "risk_note": "слишком большой объем для базовой аудитории",
        },
        {
            "dataset_id": "kaist_industrial_ac_motors",
            "name": "Industrial-Scale AC Motors under Randomized Speed/Load Variations",
            "url": "https://www.sciencedirect.com/science/article/pii/S235234092500678X",
            "object": "промышленные AC-двигатели при разных скоростях и нагрузках",
            "license": "CC BY 4.0",
            "access": "Mendeley Data, несколько архивов",
            "size_note": "более 60 ГБ",
            "format_note": "высокочастотные сигналы",
            "lessons": "3",
            "base_usage": "только как источник для продвинутого проекта",
            "risk_level": "высокий",
            "risk_note": "чрезмерный объем; сложная структура архивов",
        },
        {
            "dataset_id": "uci_servo",
            "name": "Servo",
            "url": "https://archive.ics.uci.edu/dataset/87/servo",
            "object": "сервомеханизм",
            "license": "CC BY 4.0",
            "access": "UCI",
            "size_note": "167 строк",
            "format_note": "CSV",
            "lessons": "2,3",
            "base_usage": "очень малый резервный пример для регрессии",
            "risk_level": "низкий",
            "risk_note": "устаревший и малый набор, ограниченная инженерная глубина",
        },
        {
            "dataset_id": "uci_electrical_grid_stability",
            "name": "Electrical Grid Stability Simulated Data",
            "url": "https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data",
            "object": "симулированная устойчивость энергосистемы",
            "license": "CC BY 4.0",
            "access": "UCI",
            "size_note": "10 000 строк, около 2.3 МБ",
            "format_note": "CSV",
            "lessons": "1,3",
            "base_usage": "резервный пример классификации устойчивости",
            "risk_level": "средний",
            "risk_note": "симуляция энергосистемы, а не двигатель",
        },
        {
            "dataset_id": "uci_combined_cycle_power_plant",
            "name": "Combined Cycle Power Plant",
            "url": "https://archive.ics.uci.edu/ml/datasets/combined%2Bcycle%2Bpower%2Bplant",
            "object": "парогазовая электростанция",
            "license": "CC BY 4.0",
            "access": "UCI",
            "size_note": "9 568 строк, около 3.5 МБ",
            "format_note": "XLSX / ODS / UCI API",
            "lessons": "2",
            "base_usage": "резервная энергетическая регрессия",
            "risk_level": "низкий",
            "risk_note": "объект не является электромеханическим преобразователем",
        },
        {
            "dataset_id": "uci_gas_turbine_emissions",
            "name": "Gas Turbine CO and NOx Emission Data Set",
            "url": "https://archive.ics.uci.edu/dataset/551/gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset",
            "object": "газовая турбина",
            "license": "CC BY 4.0",
            "access": "UCI",
            "size_note": "36 733 строк, около 1 МБ",
            "format_note": "CSV",
            "lessons": "2,3",
            "base_usage": "энергетическая регрессия и производная классификация по выбросам",
            "risk_level": "средний",
            "risk_note": "необходимо сохранять временное разделение по годам",
        },
        {
            "dataset_id": "uci_energy_efficiency",
            "name": "Energy Efficiency",
            "url": "https://archive.ics.uci.edu/dataset/242/energy%2Befficiency",
            "object": "энергетическая модель здания",
            "license": "CC BY 4.0",
            "access": "UCI",
            "size_note": "768 строк",
            "format_note": "XLSX / UCI API",
            "lessons": "2",
            "base_usage": "малый резервный пример регрессии",
            "risk_level": "низкий",
            "risk_note": "далек от электромеханики; симуляционный набор",
        },
        {
            "dataset_id": "uci_appliances_energy",
            "name": "Appliances Energy Prediction",
            "url": "https://archive.ics.uci.edu/ml/datasets/Appliances%2Benergy%2Bprediction",
            "object": "энергопотребление бытовых приборов",
            "license": "CC BY 4.0",
            "access": "UCI",
            "size_note": "19 735 строк, около 11.4 МБ",
            "format_note": "CSV",
            "lessons": "2",
            "base_usage": "резервная временная энергетическая регрессия",
            "risk_level": "средний",
            "risk_note": "временная зависимость; объект бытовой энергетики",
        },
        {
            "dataset_id": "uci_micro_gas_turbine",
            "name": "Micro Gas Turbine Electrical Energy Prediction",
            "url": "https://archive.ics.uci.edu/dataset/994",
            "object": "микрогазовая турбина 3 кВт",
            "license": "CC BY 4.0",
            "access": "UCI",
            "size_note": "71 225 строк, около 887.5 КБ",
            "format_note": "CSV / UCI API",
            "lessons": "2",
            "base_usage": "компактная динамическая регрессия управляющего воздействия и мощности",
            "risk_level": "средний",
            "risk_note": "требуются лаговые признаки для корректной динамической модели",
        },
        {
            "dataset_id": "diagnostics_interturn_short_circuits_pmsm",
            "name": "Diagnostics of Interturn Short Circuits in PMSMs",
            "url": "https://zenodo.org/records/15631383",
            "object": "PMSM с межвитковыми короткими замыканиями",
            "license": "CC BY 4.0",
            "access": "прямая загрузка Zenodo",
            "size_note": "около 91.5 МБ по данным карточки источника",
            "format_note": "экспериментальные файлы",
            "lessons": "3",
            "base_usage": "продвинутый источник диагностики коротких замыканий",
            "risk_level": "высокий",
            "risk_note": "требуется предварительное чтение MAT-файлов и оконная агрегация сигналов",
        },
        {
            "dataset_id": "pmsm_geometry_torque_zenodo",
            "name": "PMSM geometrical parameters and torque signals",
            "url": "https://zenodo.org/records/15688397",
            "object": "геометрия PMSM и сигналы момента",
            "license": "GPLv3+",
            "access": "прямая загрузка Zenodo",
            "size_note": "около 15.6 МБ по данным карточки источника",
            "format_note": "табличные и сигналовые файлы",
            "lessons": "2",
            "base_usage": "резерв для регрессии момента",
            "risk_level": "средний",
            "risk_note": "лицензия GPLv3+ требует осторожности при производных материалах",
        },
    ]

    implementation_status = {
        "zenodo_motor_temperature": "external_notebook_ready",
        "pmsm_inverter_fault_zenodo": "external_notebook_ready",
        "ev_powertrain_efficiency": "external_notebook_ready",
    }
    result = pd.DataFrame(rows)
    result["implementation_status"] = result["dataset_id"].map(implementation_status).fillna("methodology_only")
    return result


def _minimum_working_subset(dataset_id: str, format_note: str, size_note: str) -> str:
    """Вернуть конкретное ограничение объема для аудиторной работы."""

    subset_by_dataset = {
        "zenodo_motor_temperature": (
            "Использовать компактный CSV из `data/processed/external`; при чтении исходного .ts-файла "
            "сохранять profile_id и не перемешивать соседние временные фрагменты до разбиения."
        ),
        "permanent_magnet_dc_worm_motor": (
            "Выбрать 2-4 листа Excel с простыми переходными процессами; преобразовать их в один CSV "
            "с полями motor_id, trial_id и time_s."
        ),
        "uci_ai4i_2020": "Использовать полный CSV: 10 000 строк достаточно для Google Colab.",
        "figshare_three_phase_induction_fault": (
            "Не загружать полный архив на занятии; выбрать 1 нормальный и 1 отказный CSV, затем построить "
            "не более 400 окон сигналов."
        ),
        "paderborn_bearing": (
            "Выбрать 2 исправных и 2 поврежденных состояния, по 2 режима нагрузки; формировать окна сигналов "
            "или использовать заранее подготовленные агрегаты."
        ),
        "mendeley_rotating_machine_fault": (
            "Выбрать по одному файлу нормального режима и каждого дефекта; построить не более 500 окон "
            "с агрегированными признаками."
        ),
        "mcc5_thu_motor": (
            "Полный объем не использовать; выбрать не более 3 классов отказов и не более 300 окон на класс."
        ),
        "kaist_industrial_ac_motors": (
            "Использовать только заранее подготовленную таблицу признаков; полный архив более 60 ГБ "
            "непригоден для базового занятия."
        ),
        "diagnostics_interturn_short_circuits_pmsm": (
            "Выбрать 4-6 MAT-файлов из записи Zenodo; построить окна сигналов с RMS, максимумом, средним "
            "и стандартным отклонением."
        ),
        "pmsm_geometry_torque_zenodo": (
            "Выбрать табличную часть с геометрическими параметрами; если момент задан сигналом, рассчитать "
            "средний момент и амплитуду пульсаций."
        ),
    }
    if dataset_id in subset_by_dataset:
        return subset_by_dataset[dataset_id]
    if any(token in format_note.lower() for token in ["mat", "tdms", "сигнал"]):
        return "Использовать малое подмножество: не более 500 окон сигналов с агрегированными признаками."
    if "ГБ" in size_note or "GB" in size_note:
        return "Полный набор не загружать в аудитории; использовать заранее подготовленное подмножество."
    return "Для базовой работы использовать полный файл, если он загружается менее чем за 2-3 минуты."


def _split_rule(dataset_id: str, lesson: str) -> str:
    """Вернуть правило разбиения данных без утечки между обучением и проверкой."""

    if dataset_id == "zenodo_motor_temperature":
        return "Разбивать по profile_id или по временным блокам; случайное перемешивание соседних фрагментов не применять."
    if dataset_id in {
        "figshare_three_phase_induction_fault",
        "paderborn_bearing",
        "mendeley_rotating_machine_fault",
        "cwru_bearing",
        "university_ottawa_uoemd",
        "pmsm_stator_faults",
        "mcc5_thu_motor",
        "kaist_industrial_ac_motors",
        "diagnostics_interturn_short_circuits_pmsm",
    }:
        return "Окна одного исходного файла, запуска, двигателя или подшипника не делить между обучением и тестом."
    if dataset_id in {"uci_gas_turbine_emissions", "uci_appliances_energy", "uci_micro_gas_turbine"}:
        return "Использовать временное разбиение; будущие наблюдения не должны попадать в обучение при прогнозе прошлого."
    if lesson == "1":
        return "Разбиение на обучение и тест не требуется; порядок строк сохранять до первичного анализа."
    return "Использовать фиксированное train/test-разбиение 75/25 с random_state; для дисбаланса классов применять stratify."


def _target_rule(dataset_id: str, lesson: str) -> str:
    """Вернуть конкретную постановку целевой переменной для расширенного задания."""

    regression_targets = {
        "zenodo_motor_temperature": "Цель: целевая температура временного фрагмента; саму целевую температуру исключить из признаков.",
        "ev_powertrain_efficiency": "Цель: расчетный КПД силовой установки; признаки, прямо входящие в формулу КПД, использовать только после проверки утечки.",
        "permanent_magnet_dc_worm_motor": "Цель: установившаяся скорость или средний ток по окну переходного процесса.",
        "uci_combined_cycle_power_plant": "Цель: чистая электрическая мощность PE, МВт.",
        "uci_gas_turbine_emissions": "Цель: выбросы CO или NOx; разбиение выполнять по годам.",
        "uci_energy_efficiency": "Цель: нагрузка отопления или охлаждения здания.",
        "uci_micro_gas_turbine": "Цель: выходная электрическая мощность; для динамической модели использовать лаговые признаки.",
        "pmsm_geometry_torque_zenodo": "Цель: средний электромагнитный момент или амплитуда пульсаций момента.",
    }
    classification_targets = {
        "zenodo_motor_temperature": "Класс: 1, если целевая температура ниже заранее заданного учебного предела; 0, если предел превышен.",
        "uci_ai4i_2020": "Класс: Machine failure; флаги причин отказа TWF, HDF, PWF, OSF, RNF исключить из признаков.",
        "uci_electrical_grid_stability": "Класс: stabf; непрерывный показатель stab исключить из признаков.",
        "figshare_three_phase_induction_fault": "Класс: исправный или отказный режим по имени исходного файла; имя файла не включать в признаки.",
        "paderborn_bearing": "Класс: исправный или поврежденный подшипник; окна одного испытания держать в одной части разбиения.",
        "diagnostics_interturn_short_circuits_pmsm": "Класс: наличие межвиткового короткого замыкания или степень дефекта по метаданным файла.",
    }
    if lesson == "2":
        return regression_targets.get(dataset_id, "Цель: выбрать одну непрерывную физическую величину и обосновать ее доступность.")
    if lesson == "3":
        return classification_targets.get(dataset_id, "Класс: сформировать бинарную метку по заранее заданному учебному порогу.")
    return "Целевая переменная может не задаваться; требуется описать возможные цели для последующих занятий."


def _success_criteria(lesson: str) -> str:
    """Вернуть минимальные критерии успешного выполнения расширенного задания."""

    if lesson == "1":
        return "Зачет: паспорт данных, таблица пропусков, не менее 3 графиков, вывод о пригодности источника."
    if lesson == "2":
        return "Зачет: корректная цель, исключение утечек, базовая модель, MAE/RMSE/R2, график остатков."
    return "Зачет: распределение классов, метрики по опасному классу, матрица ошибок, 2-3 правила дерева."


def dataset_assignments() -> pd.DataFrame:
    """Сформировать развернутые задания по всем найденным наборам данных."""

    catalog = dataset_catalog()
    rows: list[dict[str, str]] = []

    for row in catalog.to_dict(orient="records"):
        lessons = str(row["lessons"])
        dataset_id = str(row["dataset_id"])
        object_name = str(row["object"])
        risk_note = str(row["risk_note"])
        base_usage = str(row["base_usage"])
        minimum_working_subset = _minimum_working_subset(
            dataset_id,
            str(row["format_note"]),
            str(row["size_note"]),
        )
        implementation_status = str(row["implementation_status"])
        dataset_structure = (
            f"Объект: {object_name}. Формат данных: {row['format_note']}. "
            f"Размер или масштаб: {row['size_note']}. Доступ: {row['access']}. "
            f"Лицензия или ограничение использования: {row['license']}. "
            f"Статус реализации: {implementation_status}."
        )

        if "1" in lessons:
            rows.append(
                {
                    "assignment_id": f"{dataset_id}_eda",
                    "dataset_id": dataset_id,
                    "lesson": "1",
                    "assignment_title": f"Первичный анализ данных: {row['name']}",
                    "implementation_status": implementation_status,
                    "theory_block": (
                        "Теоретический блок должен раскрыть структуру инженерного "
                        "набора данных. Необходимо определить, что является наблюдением "
                        "в выбранном источнике: строка таблицы, измерительный профиль, "
                        "экспериментальный запуск или окно временного сигнала. Далее "
                        "следует выделить физические признаки объекта, объяснить их "
                        "единицы измерения и указать, какие величины могут выступать "
                        "целевыми переменными. Отдельно требуется объяснить понятия "
                        "пропуска, выброса, временной зависимости и утечки данных "
                        "(data leakage)."
                    ),
                    "practice_block": (
                        "Практический блок включает загрузку полного набора или "
                        "подготовленного подмножества, построение паспорта столбцов, "
                        "расчет числа наблюдений, числа признаков и доли пропусков. "
                        "Необходимо выделить числовые, категориальные и служебные "
                        "столбцы, проверить диапазоны физических величин и построить "
                        "кандидаты в выбросы методом межквартильного размаха. Если "
                        "источник является временным рядом, требуется сохранить порядок "
                        "наблюдений и не перемешивать строки до анализа."
                    ),
                    "expected_artifacts": (
                        "Паспорт набора данных, таблица признаков с единицами "
                        "измерения, таблица пропусков, таблица описательной статистики, "
                        "не менее трех визуализаций и краткий вывод о пригодности "
                        "источника для инженерного анализа."
                    ),
                    "dataset_structure": dataset_structure,
                    "minimum_working_subset": minimum_working_subset,
                    "target_rule": _target_rule(dataset_id, "1"),
                    "split_rule": _split_rule(dataset_id, "1"),
                    "success_criteria": _success_criteria("1"),
                    "recommended_visualizations": (
                        "1. Гистограммы ключевых физических признаков. 2. Диаграмма "
                        "размаха (boxplot) для поиска кандидатов в выбросы. "
                        "3. Диаграмма рассеяния двух физически связанных величин. "
                        "4. Тепловая карта корреляций для числовых признаков. "
                        "5. Для временного ряда - график признака по времени или "
                        "по номеру наблюдения."
                    ),
                    "control_questions": (
                        "Что является наблюдением в данном источнике? Какие признаки "
                        "имеют физические единицы измерения? Какие столбцы нельзя "
                        "использовать как входные признаки без риска утечки данных? "
                        "Какие выбросы могут быть реальными аварийными или переходными "
                        "режимами?"
                    ),
                    "risk_note": risk_note,
                    "methodical_note": base_usage,
                }
            )

        if "2" in lessons:
            rows.append(
                {
                    "assignment_id": f"{dataset_id}_regression",
                    "dataset_id": dataset_id,
                    "lesson": "2",
                    "assignment_title": f"Регрессионная модель: {row['name']}",
                    "implementation_status": implementation_status,
                    "theory_block": (
                        "Теоретический блок должен объяснить постановку регрессии "
                        "(regression) как задачи прогнозирования непрерывной величины. "
                        "Необходимо обосновать выбор целевой переменной: температуры, "
                        "мощности, скорости, момента, КПД или энергетического показателя. "
                        "Следует указать, какие признаки доступны до момента прогноза, "
                        "а какие являются производными от целевой переменной и могут "
                        "создать утечку данных. Требуется пояснить смысл обучающей и "
                        "тестовой выборок, масштабирования признаков, регуляризации "
                        "(regularization) и остатков модели."
                    ),
                    "practice_block": (
                        "Практический блок включает выбор целевой переменной, "
                        "исключение признаков с утечкой, подготовку обучающей и "
                        "тестовой выборок, обучение базовой линейной регрессии и "
                        "более гибкой модели. Для табличных данных рекомендуется "
                        "полиномиальная гребневая регрессия (Ridge regression); для временных данных перед "
                        "моделированием следует обсудить разбиение по профилям или "
                        "по времени. После обучения нужно рассчитать MAE, RMSE и R2, "
                        "построить график фактических и прогнозных значений, график "
                        "остатков и график сравнения моделей."
                    ),
                    "expected_artifacts": (
                        "Таблица метрик качества, график сравнения моделей, график "
                        "фактических и прогнозных значений, график остатков, вывод "
                        "о физической правдоподобности прогноза и перечень признаков, "
                        "исключенных из-за риска утечки данных."
                    ),
                    "dataset_structure": dataset_structure,
                    "minimum_working_subset": minimum_working_subset,
                    "target_rule": _target_rule(dataset_id, "2"),
                    "split_rule": _split_rule(dataset_id, "2"),
                    "success_criteria": _success_criteria("2"),
                    "recommended_visualizations": (
                        "1. Распределение целевой переменной. 2. Диаграммы рассеяния "
                        "целевой переменной с основными признаками. 3. График "
                        "фактических и прогнозных значений. 4. График остатков. "
                        "5. Столбчатая диаграмма метрик качества для нескольких моделей. "
                        "6. По возможности - график зависимости качества от сложности "
                        "модели."
                    ),
                    "control_questions": (
                        "Почему выбранная цель является непрерывной величиной? Какие "
                        "признаки доступны до момента прогноза? Что показывает остаток "
                        "модели? Почему высокая метрика R2 не доказывает физическую "
                        "истинность модели?"
                    ),
                    "risk_note": risk_note,
                    "methodical_note": base_usage,
                }
            )

        if "3" in lessons:
            rows.append(
                {
                    "assignment_id": f"{dataset_id}_classification",
                    "dataset_id": dataset_id,
                    "lesson": "3",
                    "assignment_title": f"Классификация режимов: {row['name']}",
                    "implementation_status": implementation_status,
                    "theory_block": (
                        "Теоретический блок должен объяснить классификацию "
                        "(classification) как задачу отнесения наблюдения к одному из "
                        "заранее заданных классов. Необходимо обосновать целевую "
                        "переменную: исправный или неисправный режим, допустимый или "
                        "недопустимый тепловой режим, устойчивое или неустойчивое "
                        "состояние. Следует объяснить дерево решений, критерий "
                        "разделения узла, глубину дерева, матрицу ошибок, долю "
                        "правильных ответов (accuracy), точность положительных "
                        "предсказаний (precision), полноту (recall) и F1-меру. "
                        "Отдельно нужно обсудить наиболее опасную "
                        "ошибку для инженерной безопасности."
                    ),
                    "practice_block": (
                        "Практический блок включает формирование целевой переменной, "
                        "проверку распределения классов и исключение признаков с "
                        "утечкой. Если исходные данные являются временными сигналами, "
                        "сначала требуется рассчитать агрегированные признаки по "
                        "окнам: среднее значение, RMS, максимум, стандартное "
                        "отклонение, коэффициент формы или спектральный показатель. "
                        "Затем необходимо обучить дерево решений с ограничением "
                        "глубины, построить матрицу ошибок, таблицу метрик, диаграмму "
                        "важности признаков и выписать 2-3 правила в инженерной форме."
                    ),
                    "expected_artifacts": (
                        "Распределение классов, график признаков с раскраской по "
                        "классам, матрица ошибок, таблица метрик, визуализация дерева "
                        "решений, таблица важности признаков и инженерная интерпретация "
                        "наиболее опасного типа ошибки."
                    ),
                    "dataset_structure": dataset_structure,
                    "minimum_working_subset": minimum_working_subset,
                    "target_rule": _target_rule(dataset_id, "3"),
                    "split_rule": _split_rule(dataset_id, "3"),
                    "success_criteria": _success_criteria("3"),
                    "recommended_visualizations": (
                        "1. Столбчатая диаграмма распределения классов. 2. Диаграмма "
                        "рассеяния двух ключевых признаков с цветом класса. 3. Матрица "
                        "ошибок. 4. Визуализация дерева решений. 5. Диаграмма важности "
                        "признаков. 6. Для сигналов - график исходного фрагмента и "
                        "таблица агрегированных признаков."
                    ),
                    "control_questions": (
                        "Как сформирована целевая переменная? Какой тип ошибки наиболее "
                        "опасен инженерно? Какие признаки могут привести к утечке "
                        "данных? Почему дерево большой глубины может переобучиться?"
                    ),
                    "risk_note": risk_note,
                    "methodical_note": base_usage,
                }
            )

    return pd.DataFrame(rows)


def _dataset_metadata_text() -> str:
    """Вернуть методическое описание сформированных CSV-файлов."""

    return """# Учебные наборы данных для практических занятий 1-3

Дата формирования: 2026-05-07.

## Общий принцип

Таблицы являются синтетическими учебными данными, построенными на физических
соотношениях для электромеханического преобразователя:

1. `output_power_w = torque_nm * omega_rad_s`;
2. `input_power_w = voltage_v * current_a`;
3. `efficiency = output_power_w / input_power_w`;
4. `loss_power_w = input_power_w - output_power_w`;
5. температура обмотки увеличивается при росте потерь и тока.

Открытые реальные ориентиры для структуры признаков: Zenodo
`ElectricMotorTemperature` из TSML Archive, Zenodo PMSM inverter fault
diagnosis и Mendeley Data `Processed Data for EV Powertrain Efficiency`.
Они используются в расширенных блокнотах занятий 1-3 как отдельные
развернутые задания. Для обязательных базовых занятий сохраняются малые
учебные CSV-файлы, чтобы запуск не зависел от пропускной способности сети и
размера исходных архивов.

## Файлы

1. `practice_01_motor_measurements.csv` - первичный анализ данных, пропуски и
   выбросы сохранены намеренно.
2. `practice_02_motor_efficiency_features.csv` - безопасная учебная таблица
   для регрессии КПД. Столбцы `output_power_w`, `loss_power_w` и `current_a`
   вынесены из базового набора, чтобы не превращать задачу в восстановление
   физического тождества.
3. `practice_02_motor_efficiency_diagnostics.csv` - диагностические и
   расчетные столбцы занятия 2. Они используются для объяснения утечек данных,
   но не являются базовыми признаками модели.
4. `practice_03_drive_mode_features.csv` - безопасная учебная таблица для
   бинарной классификации режима электропривода, целевая переменная
   `is_allowed`: 1 - допустимый режим, 0 - недопустимый режим.
5. `practice_03_drive_mode_diagnostics.csv` - диагностические столбцы правила
   разметки занятия 3: причины нарушений, запасы до ограничений и текстовая
   причина. Эти столбцы не должны включаться в признаки базовой модели.
6. `practice_02_motor_efficiency.csv` и
   `practice_03_drive_mode_classification.csv` - полные совместимые таблицы
   для преподавательской диагностики и обратной совместимости. В базовых
   студенческих моделях они не используются.
7. `practice_01_03_dataset_catalog.csv` - реестр найденных реальных,
   открытых и вспомогательных наборов данных для занятий 1-3.
8. `practice_01_03_dataset_assignments.csv` - развернутые задания по каждому
   найденному набору данных.

## Важное различие распределений КПД

Занятие 2 и занятие 3 используют разные учебные постановки. В занятии 2
распределение `efficiency` подобрано для регрессии КПД в широком диапазоне
режимов. В занятии 3 распределение специально изменено для классификации:
часть строк формирует сбалансированную группу `low_efficiency`, чтобы дерево
решений встретило несколько типов недопустимых режимов. Поэтому меньший
верхний предел КПД в занятии 3 является методическим свойством данных, а не
противоречием физической модели двигателя.

## Внешние открытые наборы данных

Расширенные материалы хранятся в `data/processed/external/`. Для каждого
источника подготовлены три файла: feature-CSV, diagnostics-CSV и metadata-MD.

1. `zenodo_motor_temperature_features.csv`,
   `zenodo_motor_temperature_diagnostics.csv`,
   `zenodo_motor_temperature_metadata.md`.
   Источник: Zenodo `ElectricMotorTemperature`, DOI
   `10.5281/zenodo.11235562`, лицензия CC BY 4.0.
2. `zenodo_pmsm_inverter_fault_features.csv`,
   `zenodo_pmsm_inverter_fault_diagnostics.csv`,
   `zenodo_pmsm_inverter_fault_metadata.md`.
   Источник: Zenodo PMSM inverter fault diagnosis, лицензия CC BY 4.0.
3. `mendeley_ev_powertrain_efficiency_features.csv`,
   `mendeley_ev_powertrain_efficiency_diagnostics.csv`,
   `mendeley_ev_powertrain_efficiency_metadata.md`.
   Источник: Mendeley Data `Processed Data for EV Powertrain Efficiency`,
   DOI `10.17632/kbwr2z8r3y.1`, лицензия CC BY 4.0.

Файлы `external_dataset_index.csv` и `external_dataset_index.json` являются
машиночитаемым реестром внешних источников. В расширенных заданиях запрещено
использовать diagnostics-CSV как автоматический вход модели без отдельного
обоснования, поскольку в нем могут находиться производные признаки,
раскрывающие способ расчета целевой переменной.

Внешние feature-CSV намеренно не содержат целевые или прокси-целевые
столбцы, такие как `target_temperature_c`, `motor_efficiency`,
`drivetrain_efficiency` и `max_bridge_temp_c`. Эти величины хранятся в
diagnostics-CSV и присоединяются в блокнотах только для постановки цели,
интерпретации ошибок и демонстрации утечки данных.

## Ограничения применимости

Данные не предназначены для проектирования реальной электрической машины,
подтверждения паспортных характеристик, оценки безопасности или сертификации.
Их назначение - обучение воспроизводимой процедуре анализа инженерных данных.
"""

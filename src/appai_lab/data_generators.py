"""Генераторы учебных инженерных наборов данных.

Модуль используется в практических занятиях 1-9. Он формирует небольшие
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
    practice_04: Path
    practice_04_features: Path
    practice_04_diagnostics: Path
    practice_05: Path
    practice_05_features: Path
    practice_05_diagnostics: Path
    practice_06: Path
    practice_06_features: Path
    practice_06_diagnostics: Path
    practice_07_features: Path
    practice_07_diagnostics: Path
    practice_07_waveforms: Path
    practice_08_features: Path
    practice_08_diagnostics: Path
    practice_08_scenarios: Path
    practice_09: Path
    practice_09_features: Path
    practice_09_diagnostics: Path
    catalog: Path
    assignments: Path
    catalog_04_06: Path
    assignments_04_06: Path
    catalog_07_09: Path
    assignments_07_09: Path
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


def generate_haps_thermal_dataset(
    profile_count: int = 6,
    steps_per_profile: int = 120,
    random_state: int = RANDOM_SEED + 4,
) -> pd.DataFrame:
    """Сформировать учебный набор для теплового моделирования электропривода.

    HAPS (High Altitude Platform Station) - высотная псевдоспутниковая
    платформа. В учебной постановке рассматривается электропривод воздушного
    винта, работающий при меняющейся высоте, плотности воздуха и охлаждении.
    Температура обмотки рассчитывается рекуррентной тепловой моделью первого
    порядка, а затем дополняется небольшим измерительным шумом.
    """

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    dt_s = 5.0
    sample_id = 1

    altitude_centers = np.linspace(2_000.0, 19_000.0, profile_count)
    load_centers = np.array([0.35, 0.50, 0.68, 0.82, 0.60, 0.45])[:profile_count]
    segment_names = ["climb", "cruise_low", "cruise_high", "payload", "descent", "loiter"][:profile_count]

    for profile_index in range(profile_count):
        altitude_base = altitude_centers[profile_index]
        load_base = load_centers[profile_index]
        temperature_state = rng.uniform(24.0, 38.0)
        for step in range(steps_per_profile):
            time_s = step * dt_s
            mission_phase = 2.0 * np.pi * step / max(steps_per_profile - 1, 1)
            altitude_m = altitude_base + 450.0 * np.sin(mission_phase) + rng.normal(0.0, 90.0)
            altitude_m = float(np.clip(altitude_m, 500.0, 21_000.0))
            air_density_kg_m3 = 1.225 * np.exp(-altitude_m / 8_500.0)
            ambient_temp_c = 15.0 - 0.0065 * altitude_m + rng.normal(0.0, 1.2)
            ambient_temp_c = float(np.clip(ambient_temp_c, -58.0, 25.0))
            cooling_air_speed_mps = 18.0 + 34.0 * air_density_kg_m3 + rng.normal(0.0, 1.4)
            cooling_air_speed_mps = float(np.clip(cooling_air_speed_mps, 8.0, 58.0))

            load_fraction = np.clip(
                load_base + 0.10 * np.sin(2.0 * mission_phase + profile_index) + rng.normal(0.0, 0.035),
                0.20,
                0.95,
            )
            speed_rpm = 2_200.0 + 4_100.0 * load_fraction + rng.normal(0.0, 160.0)
            speed_rpm = float(np.clip(speed_rpm, 1_800.0, 6_800.0))
            torque_nm = 4.0 + 17.0 * load_fraction + rng.normal(0.0, 0.45)
            torque_nm = float(np.clip(torque_nm, 2.5, 22.0))
            voltage_v = float(rng.choice([270.0, 350.0, 540.0], p=[0.30, 0.40, 0.30]) + rng.normal(0.0, 2.0))

            omega_rad_s = 2.0 * np.pi * speed_rpm / 60.0
            output_power_w = torque_nm * omega_rad_s
            efficiency = 0.88 + 0.045 * np.exp(-((load_fraction - 0.67) / 0.24) ** 2)
            efficiency -= 0.018 * (1.0 - air_density_kg_m3 / 1.225)
            efficiency += rng.normal(0.0, 0.006)
            efficiency = float(np.clip(efficiency, 0.80, 0.94))
            input_power_w = output_power_w / efficiency
            current_a = input_power_w / voltage_v + rng.normal(0.0, 0.18)
            current_a = float(np.clip(current_a, 2.0, 62.0))
            input_power_w = voltage_v * current_a
            total_loss_w = max(input_power_w - output_power_w, 40.0)

            winding_resistance_ohm = 0.072 * (1.0 + 0.0039 * (temperature_state - 20.0))
            copper_loss_w = current_a**2 * winding_resistance_ohm
            iron_loss_w = 0.0000022 * speed_rpm**2 + 18.0 * load_fraction
            mechanical_loss_w = 0.00000085 * speed_rpm**2 + 4.0 * cooling_air_speed_mps
            total_loss_w = 0.55 * total_loss_w + 0.45 * (copper_loss_w + iron_loss_w + mechanical_loss_w)

            thermal_resistance_k_per_w = 0.95 / np.sqrt(max(cooling_air_speed_mps, 1.0))
            thermal_time_constant_s = 110.0 + 180.0 / max(air_density_kg_m3, 0.08)
            steady_state_temp_c = ambient_temp_c + thermal_resistance_k_per_w * total_loss_w
            temperature_state += (dt_s / thermal_time_constant_s) * (steady_state_temp_c - temperature_state)
            winding_temp_c = temperature_state + rng.normal(0.0, 1.1)
            thermal_limit_c = 65.0

            rows.append(
                {
                    "sample_id": sample_id,
                    "profile_id": profile_index + 1,
                    "time_s": time_s,
                    "altitude_m": altitude_m,
                    "air_density_kg_m3": air_density_kg_m3,
                    "ambient_temp_c": ambient_temp_c,
                    "cooling_air_speed_mps": cooling_air_speed_mps,
                    "speed_rpm": speed_rpm,
                    "torque_nm": torque_nm,
                    "voltage_v": voltage_v,
                    "current_a": current_a,
                    "winding_temp_c": winding_temp_c,
                    "output_power_w": output_power_w,
                    "input_power_w": input_power_w,
                    "copper_loss_w": copper_loss_w,
                    "iron_loss_w": iron_loss_w,
                    "mechanical_loss_w": mechanical_loss_w,
                    "total_loss_w": total_loss_w,
                    "thermal_resistance_k_per_w": thermal_resistance_k_per_w,
                    "thermal_time_constant_s": thermal_time_constant_s,
                    "steady_state_temp_c": steady_state_temp_c,
                    "temperature_margin_c": thermal_limit_c - winding_temp_c,
                    "is_overheated": int(winding_temp_c > thermal_limit_c),
                    "mission_segment": segment_names[profile_index],
                }
            )
            sample_id += 1

    return _round_engineering_columns(pd.DataFrame(rows))


def generate_partial_discharge_dataset(
    n_per_class: int = 160,
    random_state: int = RANDOM_SEED + 5,
) -> pd.DataFrame:
    """Сформировать PRPD-признаки для классификации частичных разрядов.

    PRPD (Phase Resolved Partial Discharge) - фазово-разрешенное
    представление частичных разрядов. Вместо сырых импульсных сигналов
    учебный CSV содержит уже извлеченные признаки: заряд, фазовое положение,
    частоту повторения и статистики формы импульса.
    """

    rng = np.random.default_rng(random_state)
    class_specs = [
        ("no_pd", 0, 8.0, 6.0, 180.0, 95.0, 1.0),
        ("corona", 1, 42.0, 28.0, 92.0, 23.0, 1.8),
        ("surface", 2, 85.0, 48.0, 140.0, 55.0, 1.2),
        ("internal", 3, 145.0, 65.0, 60.0, 32.0, 0.95),
    ]
    rows: list[dict[str, float | int | str]] = []
    sample_id = 1

    for label, code, charge_center, charge_spread, phase_center, phase_std, pn_center in class_specs:
        for _ in range(n_per_class):
            voltage_kv = rng.uniform(6.0, 24.0)
            frequency_hz = rng.choice([50.0, 60.0], p=[0.82, 0.18])
            pulse_count = int(np.clip(rng.poisson(10 + 0.11 * charge_center) + code * rng.integers(2, 10), 0, 70))
            mean_charge_pc = float(np.clip(rng.normal(charge_center, charge_spread), 0.5, 320.0))
            max_charge_pc = float(mean_charge_pc * rng.uniform(1.25, 2.60))
            phase_mean_deg = float((rng.normal(phase_center, phase_std / 3.0)) % 360.0)
            if label == "internal" and rng.random() < 0.48:
                phase_mean_deg = float((phase_mean_deg + 180.0 + rng.normal(0.0, 9.0)) % 360.0)
            phase_std_deg = float(np.clip(rng.normal(phase_std, phase_std * 0.15), 8.0, 115.0))
            charge_iqr_pc = float(np.clip(mean_charge_pc * rng.uniform(0.18, 0.55), 0.1, 170.0))
            repetition_rate_hz = float(np.clip(pulse_count * frequency_hz / 100.0 + rng.normal(0.0, 1.2), 0.0, 80.0))
            positive_negative_ratio = float(np.clip(rng.normal(pn_center, 0.22), 0.25, 3.0))
            waveform_rise_ns = float(np.clip(rng.normal(22.0 + 9.0 * code, 5.5), 5.0, 95.0))
            waveform_width_ns = float(np.clip(rng.normal(90.0 + 34.0 * code, 18.0), 25.0, 310.0))
            prpd_entropy = float(np.clip(rng.normal(0.25 + 0.17 * code, 0.06), 0.05, 0.96))
            risk_score = float(
                np.clip(
                    0.22 * code
                    + 0.0025 * mean_charge_pc
                    + 0.004 * pulse_count
                    + rng.normal(0.0, 0.04),
                    0.0,
                    1.0,
                )
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "profile_id": int(rng.integers(1, 9)),
                    "voltage_kv": voltage_kv,
                    "frequency_hz": frequency_hz,
                    "phase_mean_deg": phase_mean_deg,
                    "phase_std_deg": phase_std_deg,
                    "pulse_count": pulse_count,
                    "mean_charge_pc": mean_charge_pc,
                    "max_charge_pc": max_charge_pc,
                    "charge_iqr_pc": charge_iqr_pc,
                    "repetition_rate_hz": repetition_rate_hz,
                    "positive_negative_ratio": positive_negative_ratio,
                    "waveform_rise_ns": waveform_rise_ns,
                    "waveform_width_ns": waveform_width_ns,
                    "prpd_entropy": prpd_entropy,
                    "defect_class": label,
                    "defect_code": code,
                    "has_partial_discharge": int(label != "no_pd"),
                    "risk_score": risk_score,
                    "high_risk": int(risk_score >= 0.62),
                    "phase_cluster_low_deg": float((phase_mean_deg - phase_std_deg) % 360.0),
                    "phase_cluster_high_deg": float((phase_mean_deg + phase_std_deg) % 360.0),
                    "noise_floor_pc": float(rng.uniform(0.4, 4.5)),
                }
            )
            sample_id += 1

    result = pd.DataFrame(rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    result["sample_id"] = np.arange(1, len(result) + 1)
    return _round_engineering_columns(result)


def generate_equipment_modes_dataset(
    random_state: int = RANDOM_SEED + 6,
) -> pd.DataFrame:
    """Сформировать набор режимов оборудования для кластеризации.

    В feature-CSV занятия 6 нет целевой переменной. Столбцы `true_mode_label`
    и `anomaly_flag` сохраняются только в diagnostics-CSV и используются
    после кластеризации для методической интерпретации найденных групп.
    """

    rng = np.random.default_rng(random_state)
    mode_specs = [
        ("idle", 150, 1_200.0, 3.0, 6.0, 31.0, 1.1, 0.90),
        ("nominal", 150, 3_200.0, 8.0, 18.0, 48.0, 1.8, 0.88),
        ("high_load", 110, 4_800.0, 15.0, 38.0, 78.0, 3.0, 0.84),
        ("cooling_degraded", 85, 3_900.0, 10.0, 27.0, 92.0, 2.6, 0.82),
        ("vibration_alarm", 70, 2_900.0, 7.0, 16.0, 57.0, 6.5, 0.86),
        ("sensor_drift", 35, 2_400.0, 5.5, 14.0, 42.0, 2.4, 0.87),
    ]
    rows: list[dict[str, float | int | str]] = []
    sample_id = 1

    for mode_id, (label, count, speed_center, torque_center, current_center, temp_center, vibration_center, eff_center) in enumerate(mode_specs):
        for _ in range(count):
            speed_rpm = float(np.clip(rng.normal(speed_center, speed_center * 0.08), 700.0, 6_200.0))
            torque_nm = float(np.clip(rng.normal(torque_center, max(0.35, torque_center * 0.12)), 0.3, 22.0))
            voltage_v = float(rng.choice([48.0, 96.0, 300.0], p=[0.25, 0.35, 0.40]) + rng.normal(0.0, 1.2))
            current_a = float(np.clip(rng.normal(current_center, max(0.7, current_center * 0.14)), 1.0, 58.0))
            temperature_c = float(np.clip(rng.normal(temp_center, 4.5), 18.0, 125.0))
            vibration_rms_mm_s = float(np.clip(rng.normal(vibration_center, max(0.18, vibration_center * 0.18)), 0.15, 12.0))
            acoustic_db = float(np.clip(48.0 + 3.2 * vibration_rms_mm_s + rng.normal(0.0, 2.2), 38.0, 92.0))
            cooling_flow_lpm = float(np.clip(rng.normal(18.0 - 0.10 * temperature_c + 0.0012 * speed_rpm, 1.6), 2.0, 25.0))
            if label == "cooling_degraded":
                cooling_flow_lpm *= rng.uniform(0.42, 0.65)
            efficiency = float(np.clip(rng.normal(eff_center, 0.025) - 0.002 * max(vibration_rms_mm_s - 4.0, 0.0), 0.55, 0.94))
            pressure_kpa = float(np.clip(rng.normal(101.0 - 0.11 * cooling_flow_lpm + 0.035 * current_a, 1.8), 82.0, 118.0))
            output_power_w = torque_nm * (2.0 * np.pi * speed_rpm / 60.0)
            health_score = float(
                np.clip(
                    1.0
                    - 0.0035 * max(temperature_c - 70.0, 0.0)
                    - 0.055 * max(vibration_rms_mm_s - 3.5, 0.0)
                    - 0.010 * max(current_a - 35.0, 0.0),
                    0.0,
                    1.0,
                )
            )
            anomaly_flag = int(label in {"cooling_degraded", "vibration_alarm", "sensor_drift"} or health_score < 0.55)

            rows.append(
                {
                    "sample_id": sample_id,
                    "profile_id": int(rng.integers(1, 7)),
                    "speed_rpm": speed_rpm,
                    "torque_nm": torque_nm,
                    "current_a": current_a,
                    "voltage_v": voltage_v,
                    "temperature_c": temperature_c,
                    "vibration_rms_mm_s": vibration_rms_mm_s,
                    "acoustic_db": acoustic_db,
                    "cooling_flow_lpm": cooling_flow_lpm,
                    "efficiency": efficiency,
                    "pressure_kpa": pressure_kpa,
                    "output_power_w": output_power_w,
                    "true_mode_label": label,
                    "mode_id": mode_id,
                    "anomaly_flag": anomaly_flag,
                    "health_score": health_score,
                    "maintenance_priority": "high" if health_score < 0.45 else "medium" if health_score < 0.70 else "low",
                }
            )
            sample_id += 1

    result = pd.DataFrame(rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    result["sample_id"] = np.arange(1, len(result) + 1)
    return _round_engineering_columns(result)


def generate_pd_signal_analysis_dataset(
    n_per_class: int = 80,
    n_points: int = 128,
    random_state: int = RANDOM_SEED + 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Сформировать учебные сигналы частичных разрядов и признаки.

    Частичный разряд (partial discharge, PD) моделируется как короткий
    экспоненциальный импульс на фоне измерительного шума. Для занятия 7
    сохраняются две формы представления: длинная таблица отсчетов сигнала и
    компактная таблица признаков. Такой формат позволяет студенту увидеть
    переход от временного сигнала к признаковому описанию.

    Признак threshold_crossing_count является числом отсчетов, превысивших
    адаптивный порог, а не истинным числом физических импульсов. Истинное
    число смоделированных импульсов хранится только в diagnostics-CSV и не
    используется как входной признак классификатора.
    """

    rng = np.random.default_rng(random_state)
    sampling_rate_hz = 2_000_000.0
    dt_s = 1.0 / sampling_rate_hz
    time_s = np.arange(n_points) * dt_s
    class_specs = [
        ("normal", 0, 0.018, (0, 1), 0.00, 0.0),
        ("rare_pd", 1, 0.022, (1, 3), 0.33, 0.0),
        ("frequent_pd", 2, 0.026, (4, 8), 0.56, 0.0),
        ("noisy", 3, 0.060, (0, 2), 0.20, 0.0),
        ("external_interference", 4, 0.030, (0, 2), 0.10, 240_000.0),
    ]
    feature_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    waveform_rows: list[dict[str, float | int | str]] = []
    sample_id = 1

    for label, code, noise_std, pulse_range, pd_probability, interference_freq_hz in class_specs:
        for _ in range(n_per_class):
            profile_id = int(rng.integers(1, 9))
            baseline = rng.normal(0.0, noise_std, n_points)
            low_freq = 0.010 * np.sin(2.0 * np.pi * rng.uniform(8_000.0, 18_000.0) * time_s)
            signal = baseline + low_freq
            pulse_count = int(rng.integers(pulse_range[0], pulse_range[1] + 1))
            if rng.random() > pd_probability and label in {"normal", "noisy", "external_interference"}:
                pulse_count = 0

            pulse_positions: list[int] = []
            for _pulse in range(pulse_count):
                center = int(rng.integers(12, n_points - 16))
                pulse_positions.append(center)
                amplitude = rng.uniform(0.18, 0.42) * (1.0 + 0.35 * code)
                sign = rng.choice([-1.0, 1.0], p=[0.35, 0.65])
                decay = np.exp(-np.arange(0, 18) / rng.uniform(2.4, 5.2))
                end = min(n_points, center + len(decay))
                signal[center:end] += sign * amplitude * decay[: end - center]

            if interference_freq_hz > 0:
                signal += 0.16 * np.sin(2.0 * np.pi * interference_freq_hz * time_s + rng.uniform(0.0, 2.0 * np.pi))

            fft_values = np.fft.rfft(signal)
            frequencies_hz = np.fft.rfftfreq(n_points, d=dt_s)
            spectrum = np.abs(fft_values)
            spectrum[0] = 0.0
            spectrum_power = spectrum**2
            dominant_index = int(np.argmax(spectrum))
            dominant_freq_khz = frequencies_hz[dominant_index] / 1_000.0
            spectral_centroid_khz = float(
                np.sum(frequencies_hz * spectrum_power) / max(np.sum(spectrum_power), 1e-12) / 1_000.0
            )
            energy = float(np.sum(signal**2) * dt_s)
            rms = float(np.sqrt(np.mean(signal**2)))
            max_abs = float(np.max(np.abs(signal)))
            threshold = max(0.08, 4.0 * noise_std)
            threshold_crossing_count = int(np.sum(np.abs(signal) > threshold))
            noise_power = noise_std**2
            # При коротком шумном окне оценка мощности полезной компоненты
            # может стать отрицательной после вычитания шума. Нижняя граница
            # предотвращает неопределенный логарифм и соответствует очень
            # низкому отношению сигнал-шум.
            signal_power = max(float(np.mean(signal**2)) - noise_power, 1e-12)
            snr_db = float(10.0 * np.log10(signal_power / max(noise_power, 1e-12)))

            feature_rows.append(
                {
                    "sample_id": sample_id,
                    "profile_id": profile_id,
                    "sampling_rate_hz": sampling_rate_hz,
                    "window_duration_us": n_points * dt_s * 1_000_000.0,
                    "max_abs_voltage_v": max_abs,
                    "rms_voltage_v": rms,
                    "signal_energy": energy,
                    "threshold_crossing_count": threshold_crossing_count,
                    "dominant_freq_khz": dominant_freq_khz,
                    "spectral_centroid_khz": spectral_centroid_khz,
                    "snr_db": snr_db,
                    "condition_class": label,
                }
            )
            diagnostic_rows.append(
                {
                    "sample_id": sample_id,
                    "state_code": code,
                    "direct_state_label": label,
                    "true_pulse_count": pulse_count,
                    "noise_std_v": noise_std,
                    "interference_freq_khz": interference_freq_hz / 1_000.0,
                    "first_pulse_index": pulse_positions[0] if pulse_positions else -1,
                    "leakage_pd_indicator": int(label in {"rare_pd", "frequent_pd"}),
                }
            )
            for point_index, value in enumerate(signal):
                waveform_rows.append(
                    {
                        "sample_id": sample_id,
                        "profile_id": profile_id,
                        "condition_class": label,
                        "point_index": point_index,
                        "time_us": time_s[point_index] * 1_000_000.0,
                        "voltage_v": float(value),
                    }
                )
            sample_id += 1

    features = pd.DataFrame(feature_rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    id_order = {old_id: new_id for new_id, old_id in enumerate(features["sample_id"], start=1)}
    features["sample_id"] = np.arange(1, len(features) + 1)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["sample_id"] = diagnostics["sample_id"].map(id_order)
    diagnostics = diagnostics.sort_values("sample_id").reset_index(drop=True)

    waveforms = pd.DataFrame(waveform_rows)
    waveforms["sample_id"] = waveforms["sample_id"].map(id_order)
    waveforms = waveforms.sort_values(["sample_id", "point_index"]).reset_index(drop=True)

    return (
        _round_engineering_columns(features),
        _round_engineering_columns(diagnostics),
        _round_engineering_columns(waveforms),
    )


def create_power_flow_network(scenario: dict[str, float]):
    """Создать 6-узловую учебную сеть pandapower для одного сценария."""

    try:
        import pandapower as pp
    except ImportError as exc:  # pragma: no cover - проверяется окружением
        raise ImportError(
            "Для занятий 8-9 требуется pandapower. Установите requirements-extended.txt "
            "или requirements-colab.txt."
        ) from exc

    net = pp.create_empty_network(sn_mva=100.0)
    for name in ["Grid", "Bus 1", "Bus 2", "Bus 3", "Bus 4", "Bus 5"]:
        pp.create_bus(net, vn_kv=110.0, name=name)
    pp.create_ext_grid(net, bus=0, vm_pu=1.02, name="External grid")
    pp.create_gen(
        net,
        bus=3,
        p_mw=float(scenario["generator_p_mw"]),
        vm_pu=float(scenario["generator_vm_pu"]),
        name="Distributed generator",
    )
    line_specs = [
        (0, 1, 18.0),
        (1, 2, 22.0),
        (1, 3, 16.0),
        (2, 4, 20.0),
        (3, 4, 14.0),
        (4, 5, 18.0),
        (2, 5, 25.0),
    ]
    for index, (from_bus, to_bus, length_km) in enumerate(line_specs, start=1):
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=0.08,
            x_ohm_per_km=0.32,
            c_nf_per_km=11.0,
            max_i_ka=0.45,
            name=f"Line {index}",
        )
    for bus in [2, 3, 4, 5]:
        pp.create_load(
            net,
            bus=bus,
            p_mw=float(scenario[f"load_bus_{bus}_mw"]),
            q_mvar=float(scenario[f"load_bus_{bus}_mvar"]),
            name=f"Load bus {bus}",
        )
    return net


def _run_power_flow_scenario(scenario: dict[str, float]) -> dict[str, float | int | str]:
    """Выполнить AC- и DC-расчет для одного сценария сети."""

    import logging
    import pandapower as pp

    logging.getLogger("pandapower").setLevel(logging.ERROR)
    net = create_power_flow_network(scenario)
    pp.runpp(net, algorithm="nr", init="flat", numba=False)
    ac_bus_vm = net.res_bus["vm_pu"].to_numpy()
    ac_bus_va = net.res_bus["va_degree"].to_numpy()
    ac_line_p = net.res_line["p_from_mw"].to_numpy()
    ac_line_loading = net.res_line["loading_percent"].to_numpy()
    ac_losses_mw = float(net.res_line["pl_mw"].sum())

    net_dc = create_power_flow_network(scenario)
    pp.rundcpp(net_dc, numba=False)
    dc_line_p = net_dc.res_line["p_from_mw"].to_numpy()
    dc_line_loading = net_dc.res_line["loading_percent"].to_numpy()
    line_abs_error = np.abs(ac_line_p - dc_line_p)

    result: dict[str, float | int | str] = {
        "scenario_id": int(scenario["scenario_id"]),
        "scenario_label": str(scenario["scenario_label"]),
        "total_load_mw": float(sum(scenario[f"load_bus_{bus}_mw"] for bus in [2, 3, 4, 5])),
        "total_reactive_load_mvar": float(sum(scenario[f"load_bus_{bus}_mvar"] for bus in [2, 3, 4, 5])),
        "min_vm_pu": float(ac_bus_vm.min()),
        "max_line_loading_percent": float(ac_line_loading.max()),
        "total_line_loss_mw": ac_losses_mw,
        "weak_bus": int(np.argmin(ac_bus_vm)),
        "critical_line": int(np.argmax(ac_line_loading) + 1),
        "mean_abs_line_error_mw": float(line_abs_error.mean()),
        "max_abs_line_error_mw": float(line_abs_error.max()),
        # Для почти нулевых перетоков относительная ошибка теряет физический
        # смысл и становится численно неустойчивой. В учебной метрике
        # используется инженерный нижний масштаб 1 МВт.
        "mean_relative_line_error_percent": float(
            100.0 * np.mean(line_abs_error / np.maximum(np.abs(ac_line_p), 1.0))
        ),
    }
    for index, value in enumerate(ac_bus_vm):
        result[f"bus_{index}_vm_pu"] = float(value)
    for index, value in enumerate(ac_bus_va):
        result[f"bus_{index}_va_degree"] = float(value)
    for index, value in enumerate(ac_line_p, start=1):
        result[f"ac_line_{index}_p_mw"] = float(value)
    for index, value in enumerate(ac_line_loading, start=1):
        result[f"ac_line_{index}_loading_percent"] = float(value)
    for index, value in enumerate(dc_line_p, start=1):
        result[f"dc_line_{index}_p_mw"] = float(value)
    for index, value in enumerate(dc_line_loading, start=1):
        result[f"dc_line_{index}_loading_percent"] = float(value)
    for index, value in enumerate(line_abs_error, start=1):
        result[f"line_{index}_abs_error_mw"] = float(value)
    return result


def generate_power_flow_scenario_dataset(
    n_scenarios: int = 120,
    random_state: int = RANDOM_SEED + 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Сформировать сценарии и результаты расчета 6-узловой сети.

    Возвращает таблицы для занятий 8 и 9. Одна строка соответствует одному
    сценарию нагрузки и генерации. Диагностические столбцы с AC/DC-результатами
    отделяются от feature-CSV, поскольку они могут создать утечку в
    суррогатной модели занятия 9.
    """

    rng = np.random.default_rng(random_state)
    base_loads = {
        2: (32.0, 11.0),
        3: (24.0, 8.0),
        4: (28.0, 10.0),
        5: (18.0, 6.0),
    }
    scenario_rows: list[dict[str, float | int | str]] = []
    for scenario_id in range(1, n_scenarios + 1):
        load_scale = float(rng.uniform(0.72, 1.38))
        if scenario_id == 1:
            label = "base"
            load_scale = 1.0
            generator_scale = 1.0
        elif scenario_id % 5 == 0:
            label = "high_load"
            load_scale = float(rng.uniform(1.12, 1.45))
            generator_scale = float(rng.uniform(0.82, 1.05))
        elif scenario_id % 7 == 0:
            label = "low_generation"
            generator_scale = float(rng.uniform(0.55, 0.80))
        elif scenario_id % 11 == 0:
            label = "redistributed_load"
            generator_scale = float(rng.uniform(0.90, 1.20))
        else:
            label = "random_operating_point"
            generator_scale = float(rng.uniform(0.70, 1.30))

        row: dict[str, float | int | str] = {
            "scenario_id": scenario_id,
            "scenario_label": label,
            "load_scale": load_scale,
            "generator_p_mw": 35.0 * generator_scale + rng.normal(0.0, 1.2),
            "generator_vm_pu": float(rng.uniform(1.000, 1.030)),
        }
        if scenario_id == 1:
            row["generator_p_mw"] = 35.0
            row["generator_vm_pu"] = 1.01
        for bus, (p_base, q_base) in base_loads.items():
            local_factor = float(np.clip(rng.normal(1.0, 0.08), 0.78, 1.22))
            if label == "redistributed_load" and bus in {4, 5}:
                local_factor *= 1.18
            row[f"load_bus_{bus}_mw"] = max(4.0, p_base * load_scale * local_factor)
            row[f"load_bus_{bus}_mvar"] = max(1.0, q_base * load_scale * local_factor * rng.uniform(0.92, 1.12))
        scenario_rows.append(row)

    scenarios = _round_engineering_columns(pd.DataFrame(scenario_rows))
    result_rows = [_run_power_flow_scenario(row) for row in scenarios.to_dict(orient="records")]
    results = _round_engineering_columns(pd.DataFrame(result_rows))
    merged = scenarios.merge(results, on=["scenario_id", "scenario_label"], how="left", validate="one_to_one")

    practice_08_features = merged[
        [
            "scenario_id",
            "scenario_label",
            "total_load_mw",
            "total_reactive_load_mvar",
            "generator_p_mw",
            "generator_vm_pu",
            "min_vm_pu",
            "max_line_loading_percent",
            "total_line_loss_mw",
            "weak_bus",
            "critical_line",
        ]
    ].copy()
    practice_08_diagnostics = merged[
        ["scenario_id"]
        + [f"bus_{index}_vm_pu" for index in range(6)]
        + [f"bus_{index}_va_degree" for index in range(6)]
        + [f"ac_line_{index}_p_mw" for index in range(1, 8)]
        + [f"ac_line_{index}_loading_percent" for index in range(1, 8)]
    ].copy()
    practice_08_diagnostics["voltage_violation"] = (practice_08_features["min_vm_pu"] < 0.97).astype(int)
    practice_08_diagnostics["line_overload"] = (
        practice_08_features["max_line_loading_percent"] > 100.0
    ).astype(int)

    input_columns = [
        "scenario_id",
        "scenario_label",
        "load_scale",
        "generator_p_mw",
        "generator_vm_pu",
        "load_bus_2_mw",
        "load_bus_2_mvar",
        "load_bus_3_mw",
        "load_bus_3_mvar",
        "load_bus_4_mw",
        "load_bus_4_mvar",
        "load_bus_5_mw",
        "load_bus_5_mvar",
    ]
    practice_09_features = merged[
        input_columns + ["min_vm_pu", "max_line_loading_percent"]
    ].copy()
    practice_09_diagnostics = merged[
        ["scenario_id", "total_load_mw", "total_reactive_load_mvar", "total_line_loss_mw"]
        + [f"ac_line_{index}_p_mw" for index in range(1, 8)]
        + [f"dc_line_{index}_p_mw" for index in range(1, 8)]
        + [f"line_{index}_abs_error_mw" for index in range(1, 8)]
        + [
            "mean_abs_line_error_mw",
            "max_abs_line_error_mw",
            "mean_relative_line_error_percent",
            "critical_line",
            "weak_bus",
        ]
    ].copy()
    practice_09_full = practice_09_features.merge(
        practice_09_diagnostics, on="scenario_id", how="left", validate="one_to_one"
    )
    return (
        scenarios,
        _round_engineering_columns(practice_08_features),
        _round_engineering_columns(practice_08_diagnostics),
        _round_engineering_columns(practice_09_features),
        _round_engineering_columns(practice_09_diagnostics),
        _round_engineering_columns(practice_09_full),
    )


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
        "time_s": 1,
        "altitude_m": 1,
        "air_density_kg_m3": 5,
        "cooling_air_speed_mps": 3,
        "winding_temp_c": 3,
        "input_power_w": 3,
        "copper_loss_w": 3,
        "iron_loss_w": 3,
        "mechanical_loss_w": 3,
        "total_loss_w": 3,
        "thermal_resistance_k_per_w": 6,
        "thermal_time_constant_s": 3,
        "steady_state_temp_c": 3,
        "voltage_kv": 3,
        "frequency_hz": 2,
        "phase_mean_deg": 3,
        "phase_std_deg": 3,
        "mean_charge_pc": 3,
        "max_charge_pc": 3,
        "charge_iqr_pc": 3,
        "repetition_rate_hz": 3,
        "positive_negative_ratio": 4,
        "waveform_rise_ns": 3,
        "waveform_width_ns": 3,
        "prpd_entropy": 5,
        "risk_score": 5,
        "phase_cluster_low_deg": 3,
        "phase_cluster_high_deg": 3,
        "noise_floor_pc": 3,
        "vibration_rms_mm_s": 4,
        "acoustic_db": 3,
        "cooling_flow_lpm": 3,
        "pressure_kpa": 3,
        "health_score": 5,
        "sampling_rate_hz": 1,
        "window_duration_us": 3,
        "max_abs_voltage_v": 5,
        "rms_voltage_v": 5,
        "signal_energy": 10,
        "dominant_freq_khz": 3,
        "spectral_centroid_khz": 3,
        "snr_db": 3,
        "noise_std_v": 5,
        "interference_freq_khz": 3,
        "time_us": 3,
        "load_scale": 4,
        "generator_p_mw": 4,
        "generator_vm_pu": 5,
        "load_bus_2_mw": 4,
        "load_bus_2_mvar": 4,
        "load_bus_3_mw": 4,
        "load_bus_3_mvar": 4,
        "load_bus_4_mw": 4,
        "load_bus_4_mvar": 4,
        "load_bus_5_mw": 4,
        "load_bus_5_mvar": 4,
        "total_load_mw": 4,
        "total_reactive_load_mvar": 4,
        "min_vm_pu": 5,
        "max_line_loading_percent": 3,
        "total_line_loss_mw": 5,
        "mean_abs_line_error_mw": 5,
        "max_abs_line_error_mw": 5,
        "mean_relative_line_error_percent": 3,
    }
    for index in range(6):
        decimals[f"bus_{index}_vm_pu"] = 5
        decimals[f"bus_{index}_va_degree"] = 4
    for index in range(1, 8):
        decimals[f"ac_line_{index}_p_mw"] = 5
        decimals[f"ac_line_{index}_loading_percent"] = 3
        decimals[f"dc_line_{index}_p_mw"] = 5
        decimals[f"dc_line_{index}_loading_percent"] = 3
        decimals[f"line_{index}_abs_error_mw"] = 5
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


def _practice_04_feature_columns() -> list[str]:
    """Столбцы безопасной таблицы для регрессии температуры обмотки."""

    return [
        "sample_id",
        "profile_id",
        "time_s",
        "altitude_m",
        "air_density_kg_m3",
        "ambient_temp_c",
        "cooling_air_speed_mps",
        "speed_rpm",
        "torque_nm",
        "voltage_v",
        "current_a",
        "winding_temp_c",
    ]


def _practice_04_diagnostic_columns() -> list[str]:
    """Диагностические тепловые величины занятия 4."""

    return [
        "sample_id",
        "output_power_w",
        "input_power_w",
        "copper_loss_w",
        "iron_loss_w",
        "mechanical_loss_w",
        "total_loss_w",
        "thermal_resistance_k_per_w",
        "thermal_time_constant_s",
        "steady_state_temp_c",
        "temperature_margin_c",
        "is_overheated",
        "mission_segment",
    ]


def _practice_05_feature_columns() -> list[str]:
    """Столбцы учебной таблицы для классификации частичных разрядов."""

    return [
        "sample_id",
        "profile_id",
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
        "defect_class",
    ]


def _practice_05_diagnostic_columns() -> list[str]:
    """Диагностические столбцы правила разметки занятия 5."""

    return [
        "sample_id",
        "defect_code",
        "has_partial_discharge",
        "risk_score",
        "high_risk",
        "phase_cluster_low_deg",
        "phase_cluster_high_deg",
        "noise_floor_pc",
    ]


def _practice_06_feature_columns() -> list[str]:
    """Столбцы таблицы для обучения без учителя."""

    return [
        "sample_id",
        "profile_id",
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


def _practice_06_diagnostic_columns() -> list[str]:
    """Диагностические столбцы для интерпретации кластеров."""

    return [
        "sample_id",
        "output_power_w",
        "true_mode_label",
        "mode_id",
        "anomaly_flag",
        "health_score",
        "maintenance_priority",
    ]


def create_all_datasets(output_dir: str | Path) -> DatasetPaths:
    """Создать CSV-файлы для практических занятий 1-9."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    practice_01 = output_path / "practice_01_motor_measurements.csv"
    practice_02 = output_path / "practice_02_motor_efficiency.csv"
    practice_02_features = output_path / "practice_02_motor_efficiency_features.csv"
    practice_02_diagnostics = output_path / "practice_02_motor_efficiency_diagnostics.csv"
    practice_03 = output_path / "practice_03_drive_mode_classification.csv"
    practice_03_features = output_path / "practice_03_drive_mode_features.csv"
    practice_03_diagnostics = output_path / "practice_03_drive_mode_diagnostics.csv"
    practice_04 = output_path / "practice_04_haps_thermal.csv"
    practice_04_features = output_path / "practice_04_haps_thermal_features.csv"
    practice_04_diagnostics = output_path / "practice_04_haps_thermal_diagnostics.csv"
    practice_05 = output_path / "practice_05_partial_discharge.csv"
    practice_05_features = output_path / "practice_05_partial_discharge_features.csv"
    practice_05_diagnostics = output_path / "practice_05_partial_discharge_diagnostics.csv"
    practice_06 = output_path / "practice_06_equipment_modes.csv"
    practice_06_features = output_path / "practice_06_equipment_modes_features.csv"
    practice_06_diagnostics = output_path / "practice_06_equipment_modes_diagnostics.csv"
    practice_07_features = output_path / "practice_07_pd_signal_features.csv"
    practice_07_diagnostics = output_path / "practice_07_pd_signal_diagnostics.csv"
    practice_07_waveforms = output_path / "practice_07_pd_signal_waveforms.csv"
    practice_08_features = output_path / "practice_08_power_flow_features.csv"
    practice_08_diagnostics = output_path / "practice_08_power_flow_diagnostics.csv"
    practice_08_scenarios = output_path / "practice_08_power_flow_scenarios.csv"
    practice_09 = output_path / "practice_09_power_flow_comparison.csv"
    practice_09_features = output_path / "practice_09_power_flow_comparison_features.csv"
    practice_09_diagnostics = output_path / "practice_09_power_flow_comparison_diagnostics.csv"
    catalog = output_path / "practice_01_03_dataset_catalog.csv"
    assignments = output_path / "practice_01_03_dataset_assignments.csv"
    catalog_04_06 = output_path / "practice_04_06_dataset_catalog.csv"
    assignments_04_06 = output_path / "practice_04_06_dataset_assignments.csv"
    catalog_07_09 = output_path / "practice_07_09_dataset_catalog.csv"
    assignments_07_09 = output_path / "practice_07_09_dataset_assignments.csv"
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

    practice_04_df = generate_haps_thermal_dataset()
    practice_04_df.to_csv(practice_04, index=False)
    practice_04_df[_practice_04_feature_columns()].to_csv(practice_04_features, index=False)
    practice_04_df[_practice_04_diagnostic_columns()].to_csv(practice_04_diagnostics, index=False)

    practice_05_df = generate_partial_discharge_dataset()
    practice_05_df.to_csv(practice_05, index=False)
    practice_05_df[_practice_05_feature_columns()].to_csv(practice_05_features, index=False)
    practice_05_df[_practice_05_diagnostic_columns()].to_csv(practice_05_diagnostics, index=False)

    practice_06_df = generate_equipment_modes_dataset()
    practice_06_df.to_csv(practice_06, index=False)
    practice_06_df[_practice_06_feature_columns()].to_csv(practice_06_features, index=False)
    practice_06_df[_practice_06_diagnostic_columns()].to_csv(practice_06_diagnostics, index=False)

    practice_07_features_df, practice_07_diagnostics_df, practice_07_waveforms_df = (
        generate_pd_signal_analysis_dataset()
    )
    practice_07_features_df.to_csv(practice_07_features, index=False)
    practice_07_diagnostics_df.to_csv(practice_07_diagnostics, index=False)
    practice_07_waveforms_df.to_csv(practice_07_waveforms, index=False)

    (
        practice_08_scenarios_df,
        practice_08_features_df,
        practice_08_diagnostics_df,
        practice_09_features_df,
        practice_09_diagnostics_df,
        practice_09_full_df,
    ) = generate_power_flow_scenario_dataset()
    practice_08_scenarios_df.to_csv(practice_08_scenarios, index=False)
    practice_08_features_df.to_csv(practice_08_features, index=False)
    practice_08_diagnostics_df.to_csv(practice_08_diagnostics, index=False)
    practice_09_features_df.to_csv(practice_09_features, index=False)
    practice_09_diagnostics_df.to_csv(practice_09_diagnostics, index=False)
    practice_09_full_df.to_csv(practice_09, index=False)

    dataset_catalog().to_csv(catalog, index=False)
    dataset_assignments().to_csv(assignments, index=False)
    dataset_catalog_04_06().to_csv(catalog_04_06, index=False)
    dataset_assignments_04_06().to_csv(assignments_04_06, index=False)
    dataset_catalog_07_09().to_csv(catalog_07_09, index=False)
    dataset_assignments_07_09().to_csv(assignments_07_09, index=False)
    metadata.write_text(_dataset_metadata_text(), encoding="utf-8")

    return DatasetPaths(
        practice_01=practice_01,
        practice_02=practice_02,
        practice_02_features=practice_02_features,
        practice_02_diagnostics=practice_02_diagnostics,
        practice_03=practice_03,
        practice_03_features=practice_03_features,
        practice_03_diagnostics=practice_03_diagnostics,
        practice_04=practice_04,
        practice_04_features=practice_04_features,
        practice_04_diagnostics=practice_04_diagnostics,
        practice_05=practice_05,
        practice_05_features=practice_05_features,
        practice_05_diagnostics=practice_05_diagnostics,
        practice_06=practice_06,
        practice_06_features=practice_06_features,
        practice_06_diagnostics=practice_06_diagnostics,
        practice_07_features=practice_07_features,
        practice_07_diagnostics=practice_07_diagnostics,
        practice_07_waveforms=practice_07_waveforms,
        practice_08_features=practice_08_features,
        practice_08_diagnostics=practice_08_diagnostics,
        practice_08_scenarios=practice_08_scenarios,
        practice_09=practice_09,
        practice_09_features=practice_09_features,
        practice_09_diagnostics=practice_09_diagnostics,
        catalog=catalog,
        assignments=assignments,
        catalog_04_06=catalog_04_06,
        assignments_04_06=assignments_04_06,
        catalog_07_09=catalog_07_09,
        assignments_07_09=assignments_07_09,
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


def dataset_catalog_04_06() -> pd.DataFrame:
    """Вернуть реестр открытых источников для методических заданий 4-6."""

    rows = [
        {
            "dataset_id": "nasa_cmapss",
            "name": "C-MAPSS Aircraft Engine Simulator Data",
            "url": "https://data.nasa.gov/dataset/groups/c-mapss-aircraft-engine-simulator-data",
            "object": "симулированные траектории деградации авиационного турбовентиляторного двигателя",
            "license": "NASA Open Data; требуется проверка условий конкретной карточки источника",
            "access": "NASA Open Data Portal",
            "size_note": "несколько файлов траекторий; размер зависит от выбранного поднабора FD001-FD004",
            "format_note": "текстовые таблицы временных рядов",
            "lessons": "4,6",
            "base_usage": "мини-задание по временной утечке, регрессии остаточного ресурса и кластеризации режимов деградации",
            "risk_level": "средний",
            "risk_note": "нужно разделять train/test по двигателям, а не случайно по строкам; часть сенсоров анонимизирована",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-15",
        },
        {
            "dataset_id": "mendeley_pd_cables_toa",
            "name": "Partial Discharge Signals in Insulated Power Cables with Time-of-Arrival Annotations",
            "url": "https://data.mendeley.com/datasets/3mdgxv6zt7",
            "object": "временные сигналы частичных разрядов в силовых кабелях",
            "license": "Mendeley Data; требуется проверка лицензии карточки источника",
            "access": "Mendeley Data",
            "size_note": "временные ряды напряжения и аннотации времени прихода импульсов",
            "format_note": "файлы сигналов и таблицы аннотаций",
            "lessons": "5",
            "base_usage": "мини-задание по извлечению PRPD-признаков из сырых сигналов перед классификацией",
            "risk_level": "средний",
            "risk_note": "нельзя случайно перемешивать импульсы одного измерительного опыта между train и test",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-15",
        },
        {
            "dataset_id": "zenodo_pd_calibrator",
            "name": "Dataset for New Synthetic Partial Discharge Calibrator",
            "url": "https://zenodo.org/records/8436197",
            "object": "сигналы калибратора частичных разрядов",
            "license": "Zenodo; требуется проверка лицензии версии записи",
            "access": "Zenodo",
            "size_note": "архив измерительных файлов; объем зависит от версии записи",
            "format_note": "сигналы и сопроводительные таблицы",
            "lessons": "5",
            "base_usage": "дополнительное задание по сопоставлению калиброванных импульсов и извлеченных признаков",
            "risk_level": "низкий",
            "risk_note": "источник ближе к метрологической проверке, чем к классификации реальных дефектов",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-15",
        },
        {
            "dataset_id": "uci_ai4i_2020",
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
            "object": "синтетические, но промышленно мотивированные режимы оборудования и отказы",
            "license": "UCI Machine Learning Repository; требуется проверка карточки источника",
            "access": "UCI Machine Learning Repository",
            "size_note": "10 000 наблюдений по карточке UCI",
            "format_note": "CSV",
            "lessons": "6",
            "base_usage": "мини-задание по отделению сенсорных признаков от служебных кодов, классов отказов и диагностической разметки",
            "risk_level": "средний",
            "risk_note": "столбцы отказов нельзя включать в признаки кластеризации; `Product ID` является служебным кодом",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-15",
        },
        {
            "dataset_id": "nasa_pcoe_adapt",
            "name": "NASA PCoE ADAPT Diagnostic Data",
            "url": "https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository",
            "object": "диагностические данные испытательного стенда электропитания ADAPT",
            "license": "NASA PCoE; требуется проверка условий конкретного архива",
            "access": "NASA Prognostics Center of Excellence",
            "size_note": "наборы стендовых сценариев и отказов; объем зависит от выбранного архива",
            "format_note": "таблицы временных рядов и журналы событий",
            "lessons": "6",
            "base_usage": "расширение по кластеризации режимов и отделению сенсорных каналов от событий отказа",
            "risk_level": "высокий",
            "risk_note": "сложная структура файлов; требуется предварительная нормализация временных шкал и событий",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-15",
        },
    ]
    return pd.DataFrame(rows)


def dataset_assignments_04_06() -> pd.DataFrame:
    """Сформировать развернутые задания по открытым источникам для занятий 4-6."""

    catalog = dataset_catalog_04_06()
    rows: list[dict[str, str]] = []
    for row in catalog.to_dict(orient="records"):
        lessons = str(row["lessons"]).split(",")
        for lesson in lessons:
            lesson = lesson.strip()
            if lesson == "4":
                title = f"Тепловая или ресурсная регрессия: {row['name']}"
                theory = (
                    "Опишите, какая непрерывная величина может быть целевой "
                    "переменной: температура, остаточный ресурс, показатель "
                    "деградации или ошибка физической модели. Объясните риск "
                    "временной утечки, когда соседние точки одного агрегата "
                    "попадают одновременно в train и test."
                )
                practice = (
                    "Выберите подмножество сенсорных признаков, задайте правило "
                    "разбиения по агрегату или профилю, предложите базовую "
                    "регрессионную модель и метрики MAE/RMSE. Не требуется "
                    "скачивать полный архив в аудитории."
                )
                visuals = "профиль целевой переменной по времени; распределение остаточного ресурса; график остатков"
                questions = "Что является агрегатом? Почему случайное разбиение строк опасно? Какие признаки доступны до прогноза?"
            elif lesson == "5":
                title = f"Извлечение PRPD-признаков и классификация: {row['name']}"
                theory = (
                    "Опишите переход от сырого временного сигнала к признакам "
                    "частичных разрядов: амплитуда, энергия, число импульсов, "
                    "фазовое положение и кажущийся заряд. Укажите, какие "
                    "операции нужны до обучения классификатора."
                )
                practice = (
                    "Предложите минимальный набор признаков, целевую переменную "
                    "для классификации и схему разделения измерительных опытов. "
                    "Отдельно укажите, какие аннотации нельзя использовать как "
                    "обычные признаки."
                )
                visuals = "пример временного сигнала; гистограмма амплитуд; PRPD-диаграмма; матрица ошибок"
                questions = "Что такое наблюдение: импульс, окно или опыт? Какие аннотации создают утечку? Почему нужно фиксировать частоту дискретизации?"
            else:
                title = f"Кластеризация режимов и отделение диагностической разметки: {row['name']}"
                theory = (
                    "Опишите постановку обучения без учителя: сенсорные признаки "
                    "используются до обучения, а коды отказов и служебные метки "
                    "подключаются только после кластеризации для интерпретации."
                )
                practice = (
                    "Разделите столбцы на сенсорные, служебные и диагностические, "
                    "предложите k-means/PCA или DBSCAN, сформулируйте правило "
                    "инженерного описания найденных групп."
                )
                visuals = "тепловая карта корреляций; PCA-проекция; профиль кластеров; сопоставление с отказами после обучения"
                questions = "Почему отказ нельзя использовать как признак? Как выбрать число кластеров? Что означает шум DBSCAN?"

            rows.append(
                {
                    "assignment_id": f"{row['dataset_id']}_lesson_{lesson}",
                    "dataset_id": str(row["dataset_id"]),
                    "lesson": lesson,
                    "assignment_title": title,
                    "implementation_status": str(row["implementation_status"]),
                    "theory_block": theory,
                    "practice_block": practice,
                    "expected_artifacts": (
                        "Краткий паспорт источника, таблица ролей столбцов, "
                        "перечень рисков утечки данных, 2-3 рекомендуемые "
                        "визуализации и вывод о применимости источника для "
                        f"занятия {lesson}."
                    ),
                    "dataset_structure": (
                        f"Объект: {row['object']}. Формат: {row['format_note']}. "
                        f"Доступ: {row['access']}. Размер: {row['size_note']}. "
                        f"Проверено: {row['checked_at']}."
                    ),
                    "minimum_working_subset": "Для аудиторного задания достаточно описать 500-5000 строк или 3-5 агрегатов/профилей.",
                    "target_rule": "Цель определяется студентом по постановке источника; запрещено использовать производные диагностические метки как признаки.",
                    "split_rule": "При временной или агрегатной структуре использовать разбиение по агрегату, профилю или опыту, а не случайно по строкам.",
                    "success_criteria": "Зачет: корректно определены роли столбцов, указаны риски утечки, предложены метрики и визуализации.",
                    "recommended_visualizations": visuals,
                    "control_questions": questions,
                    "risk_note": str(row["risk_note"]),
                    "methodical_note": str(row["base_usage"]),
                }
            )
    return pd.DataFrame(rows)


def dataset_catalog_07_09() -> pd.DataFrame:
    """Вернуть реестр открытых источников для методических заданий 7-9."""

    rows = [
        {
            "dataset_id": "mendeley_pd_cables_toa",
            "name": "Partial Discharge Signals in Insulated Power Cables with Time-of-Arrival Annotations",
            "url": "https://data.mendeley.com/datasets/3mdgxv6zt7",
            "object": "временные сигналы частичных разрядов в кабельной изоляции",
            "license": "Mendeley Data; проверить лицензию версии перед публикацией производных материалов",
            "access": "Mendeley Data",
            "size_note": "архив сигналов и аннотаций; для аудиторной работы требуется подмножество",
            "format_note": "временные ряды и таблицы аннотаций",
            "lessons": "7",
            "base_usage": "переход от временных отсчетов к признакам: амплитуда, энергия, число пороговых превышений, SNR и спектр",
            "risk_level": "средний",
            "risk_note": "аннотации времени прихода нельзя использовать как обычные признаки без обсуждения утечки",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-16",
        },
        {
            "dataset_id": "zenodo_pd_calibrator",
            "name": "Dataset for New Synthetic Partial Discharge Calibrator",
            "url": "https://zenodo.org/records/8436197",
            "object": "сигналы калибратора частичных разрядов",
            "license": "Zenodo; проверить лицензию конкретной версии записи",
            "access": "Zenodo",
            "size_note": "измерительные файлы калибратора, объем зависит от версии",
            "format_note": "сигналы и сопроводительные таблицы",
            "lessons": "7",
            "base_usage": "проверка устойчивости амплитудных и энергетических признаков на калиброванных импульсах",
            "risk_level": "низкий",
            "risk_note": "источник ближе к метрологическому калибратору, чем к реальному дефекту изоляции",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-16",
        },
        {
            "dataset_id": "pandapower_case9",
            "name": "pandapower IEEE 9-bus example network",
            "url": "https://pandapower.readthedocs.io/en/latest/networks/power_system_test_cases.html",
            "object": "учебная тестовая электроэнергетическая сеть",
            "license": "pandapower documentation and examples; проверить условия версии",
            "access": "поставляется с pandapower",
            "size_note": "малая тестовая сеть",
            "format_note": "объект pandapower net",
            "lessons": "8,9",
            "base_usage": "сравнение собственной 6-узловой сети с типовой тестовой сетью",
            "risk_level": "низкий",
            "risk_note": "параметры сети учебные; их нельзя переносить на реальный объект без проверки",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-16",
        },
        {
            "dataset_id": "ieee_pes_test_feeders",
            "name": "IEEE PES Distribution Test Feeders",
            "url": "https://cmte.ieee.org/pes-testfeeders/",
            "object": "распределительные тестовые сети IEEE PES",
            "license": "условия IEEE PES Test Feeders требуют проверки перед распространением",
            "access": "официальный сайт IEEE PES Test Feeders",
            "size_note": "несколько тестовых фидеров разного масштаба",
            "format_note": "файлы моделей распределительных сетей",
            "lessons": "8,9",
            "base_usage": "расширение анализа напряжений и перегрузок для распределительных сетей",
            "risk_level": "средний",
            "risk_note": "DC-приближение для распределительных сетей обычно менее применимо из-за отношения R/X",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-16",
        },
        {
            "dataset_id": "matpower_cases",
            "name": "MATPOWER case archive",
            "url": "https://matpower.org/docs/ref/matpower6.0/menu6.0.html",
            "object": "тестовые сети для расчета режима и оптимизации энергосистем",
            "license": "MATPOWER; проверить условия использования выбранного case-файла",
            "access": "MATPOWER documentation",
            "size_note": "малые и средние case-файлы",
            "format_note": "MATPOWER case format",
            "lessons": "8,9",
            "base_usage": "сравнение AC/DC-расчета на стандартных тестовых сетях",
            "risk_level": "средний",
            "risk_note": "при переносе в pandapower нужно проверить базисные мощности, номера шин и типы узлов",
            "implementation_status": "methodology_only",
            "checked_at": "2026-05-16",
        },
    ]
    return pd.DataFrame(rows)


def dataset_assignments_07_09() -> pd.DataFrame:
    """Сформировать развернутые задания по открытым источникам для занятий 7-9."""

    catalog = dataset_catalog_07_09()
    rows: list[dict[str, str]] = []
    for row in catalog.to_dict(orient="records"):
        for lesson in str(row["lessons"]).split(","):
            lesson = lesson.strip()
            if lesson == "7":
                title = f"Извлечение признаков из сигналов: {row['name']}"
                theory = (
                    "Опишите частоту дискретизации, длительность окна, способ "
                    "обнаружения импульсов, расчет энергии и переход к спектру "
                    "с помощью FFT (Fast Fourier Transform, быстрое преобразование Фурье)."
                )
                practice = (
                    "Выберите 20-50 окон сигналов, рассчитайте максимум, RMS, "
                    "энергию, число пороговых превышений, доминирующую частоту и SNR. "
                    "Отдельно укажите, какие аннотации являются диагностическими."
                )
                visuals = "временной сигнал; спектр; таблица признаков; матрица ошибок классификатора"
                questions = "Почему шаг дискретизации входит в расчет энергии? Чем спектральный пик отличается от импульса во времени?"
            elif lesson == "8":
                title = f"Расчет режима по открытой тестовой сети: {row['name']}"
                theory = (
                    "Опишите состав сети: шины, линии, нагрузки, генераторы и "
                    "внешняя сеть. Укажите, какие величины задаются до расчета, "
                    "а какие являются результатами AC power flow."
                )
                practice = (
                    "Создайте или загрузите малую сеть, выполните AC power flow, "
                    "измените нагрузку на 10-30 процентов и сравните напряжения "
                    "шин и загрузку линий."
                )
                visuals = "таблица напряжений; диаграмма загрузки линий; сравнение базового и измененного режима"
                questions = "Что является входом расчета? Почему расчет режима не является заменой машинному обучению?"
            else:
                title = f"Сравнение AC/DC и суррогатной модели: {row['name']}"
                theory = (
                    "Опишите допущения DC power flow: малые углы, малые потери, "
                    "доминирование реактивного сопротивления над активным. "
                    "Сформулируйте, какие ошибки нужно измерять."
                )
                practice = (
                    "Сформируйте несколько сценариев нагрузки, сравните AC и DC "
                    "перетоки активной мощности, затем предложите безопасные "
                    "признаки для суррогатной модели без использования AC-результатов."
                )
                visuals = "AC против DC; распределение ошибок; график прогноза суррогатной модели"
                questions = "Когда DC-приближение допустимо? Почему AC-результат нельзя включать в признаки суррогатной модели?"

            rows.append(
                {
                    "assignment_id": f"{row['dataset_id']}_lesson_{lesson}",
                    "dataset_id": str(row["dataset_id"]),
                    "lesson": lesson,
                    "assignment_title": title,
                    "implementation_status": str(row["implementation_status"]),
                    "theory_block": theory,
                    "practice_block": practice,
                    "expected_artifacts": (
                        "Паспорт источника, таблица ролей столбцов или элементов, "
                        "перечень рисков утечки, не менее двух визуализаций и "
                        "инженерный вывод о применимости источника."
                    ),
                    "dataset_structure": (
                        f"Объект: {row['object']}. Формат: {row['format_note']}. "
                        f"Доступ: {row['access']}. Размер: {row['size_note']}. "
                        f"Проверено: {row['checked_at']}."
                    ),
                    "minimum_working_subset": "Для аудиторной работы использовать малый фрагмент: 20-50 сигналов или 10-30 режимных сценариев.",
                    "target_rule": "Цель выбирается по занятию; диагностические аннотации и результаты расчетов нельзя использовать как обычные признаки.",
                    "split_rule": "Сохранять измерительный опыт, временной фрагмент или расчетный сценарий целиком; не смешивать связанные строки между train и test.",
                    "success_criteria": "Зачет: корректно определены входы, результаты, диагностические столбцы, метрики и ограничения применимости.",
                    "recommended_visualizations": visuals,
                    "control_questions": questions,
                    "risk_note": str(row["risk_note"]),
                    "methodical_note": str(row["base_usage"]),
                }
            )
    return pd.DataFrame(rows)


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

    return """# Учебные наборы данных для практических занятий 1-9

Дата формирования: 2026-05-14.

## Общий принцип

Таблицы являются синтетическими учебными данными, построенными на физических
соотношениях для электромеханического преобразователя:

1. `output_power_w = torque_nm * omega_rad_s`;
2. `input_power_w = voltage_v * current_a`;
3. `efficiency = output_power_w / input_power_w`;
4. `loss_power_w = input_power_w - output_power_w`;
5. температура обмотки увеличивается при росте потерь и тока.
6. тепловой переходный процесс электропривода приближенно описывается
   моделью первого порядка;
7. PRPD-признаки частичных разрядов формируются из фазового распределения,
   кажущегося заряда и повторяемости импульсов;
8. режимы оборудования в задаче кластеризации задаются сенсорными
   профилями без целевой переменной в feature-CSV;
9. сигнал частичных разрядов рассматривается во временной и частотной
   областях;
10. расчет режима энергосистемы выполняется как инженерная физическая
   процедура, а суррогатная модель используется только как приближение
   результатов расчета.

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
9. `practice_04_haps_thermal_features.csv` - безопасная учебная таблица для
   регрессии температуры обмотки электропривода высотной платформы.
10. `practice_04_haps_thermal_diagnostics.csv` - расчетные тепловые величины:
   потери, тепловое сопротивление, постоянная времени и запас до предельной
   температуры. Эти столбцы используются для физической интерпретации и
   демонстрации утечек, но не являются базовыми признаками модели.
11. `practice_05_partial_discharge_features.csv` - PRPD-признаки для
   классификации типа частичных разрядов. PRPD (Phase Resolved Partial
   Discharge) - фазово-разрешенное представление частичных разрядов.
12. `practice_05_partial_discharge_diagnostics.csv` - диагностические
   производные величины: код класса, признак наличия частичного разряда,
   риск-оценка и границы фазовых кластеров.
13. `practice_06_equipment_modes_features.csv` - сенсорные признаки для
   кластеризации режимов оборудования без целевой переменной.
14. `practice_06_equipment_modes_diagnostics.csv` - скрытая инженерная
   разметка режимов, флаг аномалии и показатель технического состояния.
   Эти данные можно использовать только после построения кластеров для
   интерпретации результата.
15. `practice_04_06_dataset_catalog.csv` - реестр открытых источников для
   расширенных методических заданий занятий 4-6.
16. `practice_04_06_dataset_assignments.csv` - развернутые задания по
   открытому источнику: постановка, риски утечки, рекомендуемые
   визуализации и критерии успешного выполнения.
17. `practice_07_pd_signal_features.csv` - признаки окон сигналов частичных
   разрядов: амплитуда, RMS, энергия, число пороговых превышений, спектральные
   показатели и SNR.
18. `practice_07_pd_signal_diagnostics.csv` - истинные параметры генерации
   сигналов и прямые диагностические метки. Эти столбцы используются только
   для интерпретации и антипримера утечки.
19. `practice_07_pd_signal_waveforms.csv` - длинная таблица отсчетов
   временных сигналов. Одна строка соответствует одному отсчету одного окна.
20. `practice_08_power_flow_scenarios.csv` - входные сценарии нагрузки и
   генерации для 6-узловой сети 110 кВ.
21. `practice_08_power_flow_features.csv` - сводные результаты AC power flow:
   минимальное напряжение, максимальная загрузка линии, потери и критический
   элемент.
22. `practice_08_power_flow_diagnostics.csv` - детальные напряжения шин,
   углы и загрузки линий для интерпретации режима.
23. `practice_09_power_flow_comparison_features.csv` - безопасные признаки
   сценария и целевые AC-показатели для суррогатной модели.
24. `practice_09_power_flow_comparison_diagnostics.csv` - детальное сравнение
   AC- и DC-перетоков, ошибки и расчетные величины, которые нельзя включать
   во входы суррогатной модели.
25. `practice_09_power_flow_comparison.csv` - полная таблица для
   преподавательской диагностики и обратной совместимости.
26. `practice_07_09_dataset_catalog.csv` и
   `practice_07_09_dataset_assignments.csv` - реестр источников и
   методические задания для расширения занятий 7-9.

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

## Открытые источники для расширения занятий 4-6

Для самостоятельного расширения базовых практик рекомендуется рассматривать
следующие открытые источники:

1. NASA C-MAPSS Aircraft Engine Simulator Data -
   https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data.
   Источник пригоден для обсуждения прогнозирования остаточного ресурса,
   тепловых и режимных признаков авиационного двигателя.
2. Mendeley Data `Partial Discharge Signals in Insulated Power Cables with
   Time-of-Arrival Annotations` -
   https://data.mendeley.com/datasets/3mdgxv6zt7. Набор содержит временные
   сигналы частичных разрядов и ручные аннотации времени прихода импульсов.
3. UCI `AI4I 2020 Predictive Maintenance Dataset` -
   https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset.
   Набор пригоден для сравнения учебной кластеризации режимов с задачами
   предиктивного обслуживания и диагностики отказов.

## Открытые источники для расширения занятий 7-9

Для занятий 7-9 внешние источники используются как методические задания без
обязательной загрузки больших архивов:

1. Mendeley Data `Partial Discharge Signals in Insulated Power Cables with
   Time-of-Arrival Annotations` -
   https://data.mendeley.com/datasets/3mdgxv6zt7.
2. Zenodo `Dataset for New Synthetic Partial Discharge Calibrator` -
   https://zenodo.org/records/8436197.
3. pandapower IEEE test networks -
   https://pandapower.readthedocs.io/en/latest/networks/power_system_test_cases.html.
4. IEEE PES Distribution Test Feeders -
   https://cmte.ieee.org/pes-testfeeders/.
5. MATPOWER case archive -
   https://matpower.org/docs/ref/matpower6.0/menu6.0.html.
"""

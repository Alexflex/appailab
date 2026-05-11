"""Smoke-test инвариантов курса для занятий 1-3.

Скрипт выполняет быстрые проверки, которые защищают учебные материалы от
методических регрессий: утечки целевой переменной, нарушения энергетического
баланса, случайного попадания диагностических столбцов в feature-CSV и
дисбаланса причин нарушения в классификационной задаче.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
NOTEBOOKS_STUDENT = PROJECT_ROOT / "notebooks" / "student"


REGRESSION_STRICT_FEATURES = {
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "temperature_c",
    "ambient_temp_c",
}
CLASSIFICATION_STRICT_FEATURES = {
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "current_a",
    "ambient_temp_c",
    "temperature_c",
}
LEAKAGE_COLUMNS = {
    "output_power_w",
    "loss_power_w",
    "efficiency",
    "violation_count",
    "mode_label",
    "current_margin_a",
    "temperature_margin_c",
    "speed_margin_rpm",
    "overcurrent",
    "overheating",
    "overspeed",
    "low_efficiency",
    "torque_violation",
}


def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    assert path.exists(), f"Файл не найден: {path}"
    return pd.read_csv(path)


def _check_practice_01() -> None:
    df = _read_csv("practice_01_motor_measurements.csv")
    assert df.shape == (240, 13), f"Неожиданный размер занятия 1: {df.shape}"

    missing_counts = df[["torque_nm", "current_a", "temperature_c", "efficiency"]].isna().sum()
    assert (missing_counts == 6).all(), f"Ожидалось по 6 пропусков, получено: {missing_counts.to_dict()}"

    valid = df.dropna(subset=["speed_rpm", "torque_nm", "output_power_w", "voltage_v", "current_a"])
    omega_rad_s = 2.0 * np.pi * valid["speed_rpm"] / 60.0
    recomputed_output_power_w = valid["torque_nm"] * omega_rad_s
    relative_error = (
        (recomputed_output_power_w - valid["output_power_w"]).abs()
        / valid["output_power_w"].clip(lower=1e-9)
    )
    assert relative_error.max() < 1e-3, f"Исходный энергобаланс нарушен: {relative_error.max():.6f}"

    clean_df = df.copy()
    for column in ["torque_nm", "current_a", "temperature_c"]:
        clean_df[column] = clean_df[column].fillna(clean_df[column].median())

    omega_rad_s = 2.0 * np.pi * clean_df["speed_rpm"] / 60.0
    clean_df["output_power_w"] = clean_df["torque_nm"] * omega_rad_s
    input_power_w = clean_df["voltage_v"] * clean_df["current_a"]
    clean_df["loss_power_w"] = input_power_w - clean_df["output_power_w"]
    clean_df["efficiency"] = clean_df["output_power_w"] / input_power_w

    recomputed_after = clean_df["torque_nm"] * omega_rad_s
    after_error = (
        (recomputed_after - clean_df["output_power_w"]).abs()
        / clean_df["output_power_w"].clip(lower=1e-9)
    )
    assert after_error.max() < 1e-12, f"Энергобаланс после очистки нарушен: {after_error.max():.6f}"
    assert int(clean_df.isna().sum().sum()) == 0, "После очистки занятия 1 остались пропуски."


def _check_practice_02() -> None:
    features = _read_csv("practice_02_motor_efficiency_features.csv")
    diagnostics = _read_csv("practice_02_motor_efficiency_diagnostics.csv")

    assert features.shape == (300, 8), f"Неожиданный размер feature-CSV занятия 2: {features.shape}"
    assert diagnostics.shape == (300, 6), f"Неожиданный размер diagnostics-CSV занятия 2: {diagnostics.shape}"
    assert features["sample_id"].is_unique and diagnostics["sample_id"].is_unique
    assert not (LEAKAGE_COLUMNS - {"efficiency"}).intersection(features.columns), (
        "В feature-CSV занятия 2 попали диагностические столбцы: "
        f"{sorted((LEAKAGE_COLUMNS - {'efficiency'}).intersection(features.columns))}"
    )
    assert REGRESSION_STRICT_FEATURES.issubset(features.columns), "Нет строгих признаков регрессии."
    assert "current_a" not in features.columns, "Ток не должен входить в feature-CSV строгой регрессии КПД."
    assert {"current_a", "output_power_w", "loss_power_w"}.issubset(diagnostics.columns)


def _check_practice_03() -> None:
    features = _read_csv("practice_03_drive_mode_features.csv")
    diagnostics = _read_csv("practice_03_drive_mode_diagnostics.csv")

    assert features.shape == (420, 9), f"Неожиданный размер feature-CSV занятия 3: {features.shape}"
    assert diagnostics.shape == (420, 16), f"Неожиданный размер diagnostics-CSV занятия 3: {diagnostics.shape}"
    assert features["sample_id"].is_unique and diagnostics["sample_id"].is_unique

    forbidden_in_features = LEAKAGE_COLUMNS.intersection(features.columns)
    assert not forbidden_in_features, (
        "В feature-CSV занятия 3 попали диагностические столбцы: "
        f"{sorted(forbidden_in_features)}"
    )
    assert CLASSIFICATION_STRICT_FEATURES.issubset(features.columns), "Нет строгих признаков классификации."

    full = features.merge(diagnostics, on="sample_id", how="left", validate="one_to_one")
    cause_columns = ["overcurrent", "overheating", "overspeed", "low_efficiency", "torque_violation"]
    cause_counts = full[cause_columns].sum()
    assert (cause_counts >= 25).all(), f"Недостаточно примеров причин нарушения: {cause_counts.to_dict()}"
    assert int(full["is_allowed"].sum()) == 295, "Ожидалось 295 допустимых режимов."
    assert int((full[cause_columns].sum(axis=1) > 0).sum()) == 125, "Ожидалось 125 недопустимых режимов."


def _check_student_notebooks_clear_outputs() -> None:
    for path in sorted(NOTEBOOKS_STUDENT.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert not cell.get("outputs"), f"В студенческом блокноте есть output: {path.name}, cell {index}"
            assert cell.get("execution_count") is None, (
                f"В студенческом блокноте есть execution_count: {path.name}, cell {index}"
            )


def main() -> None:
    _check_practice_01()
    _check_practice_02()
    _check_practice_03()
    _check_student_notebooks_clear_outputs()
    print("Инварианты курса для занятий 1-3 соблюдены.")


if __name__ == "__main__":
    main()

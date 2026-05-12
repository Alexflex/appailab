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
EXTERNAL_DATA_DIR = DATA_DIR / "external"
NOTEBOOKS_STUDENT = PROJECT_ROOT / "notebooks" / "student"
NOTEBOOKS_EXTERNAL_STUDENT = PROJECT_ROOT / "notebooks" / "external" / "student"
NOTEBOOKS_EXTERNAL_TEACHER = PROJECT_ROOT / "notebooks" / "external" / "teacher"


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

EXTERNAL_DATASETS = {
    "zenodo_motor_temperature",
    "zenodo_pmsm_inverter_fault",
    "mendeley_ev_powertrain_efficiency",
}
READY_DATASET_IDS = {
    "zenodo_motor_temperature",
    "pmsm_inverter_fault_zenodo",
    "ev_powertrain_efficiency",
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
        notebook_source = "\n".join(str(cell.get("source", "")) for cell in notebook.cells)
        if path.name == "03_drive_decision_tree_student.ipynb":
            assert "mode_label" in notebook_source and "not_allowed_by_mode" in notebook_source, (
                "В занятии 3 нет анализа опасных ошибок по mode_label."
            )
        if path.name == "01_engineering_data_student.ipynb":
            assert "Предупреждение о пересчете целевой переменной" in notebook_source, (
                "В занятии 1 нет предупреждения о пересчете efficiency после imputation."
            )
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert not cell.get("outputs"), f"В студенческом блокноте есть output: {path.name}, cell {index}"
            assert cell.get("execution_count") is None, (
                f"В студенческом блокноте есть execution_count: {path.name}, cell {index}"
            )


def _check_dataset_catalog_status() -> None:
    catalog = _read_csv("practice_01_03_dataset_catalog.csv")
    assignments = _read_csv("practice_01_03_dataset_assignments.csv")
    assert "implementation_status" in catalog.columns, "В каталоге нет implementation_status."
    assert "implementation_status" in assignments.columns, "В assignments нет implementation_status."
    ready_rows = catalog[catalog["implementation_status"] == "external_notebook_ready"]
    assert set(ready_rows["dataset_id"]) == READY_DATASET_IDS, (
        "Неверный список реально реализованных внешних источников: "
        f"{sorted(ready_rows['dataset_id'].tolist())}"
    )
    assert assignments["implementation_status"].isin(
        {"external_notebook_ready", "methodology_only", "planned"}
    ).all(), "В assignments найден неизвестный implementation_status."


def _check_external_datasets() -> None:
    index_path = EXTERNAL_DATA_DIR / "external_dataset_index.csv"
    assert index_path.exists(), f"Не найден индекс внешних наборов данных: {index_path}"
    index = pd.read_csv(index_path)
    assert set(index["dataset_id"]) == EXTERNAL_DATASETS, (
        "Индекс внешних наборов данных не соответствует утвержденному списку: "
        f"{sorted(index['dataset_id'].tolist())}"
    )
    assert index["prepared"].all(), "Не все внешние наборы данных подготовлены."
    assert not index["dataset_id"].str.contains("kaggle", case=False).any(), (
        "В индексе внешних данных осталась зависимость от Kaggle."
    )
    assert not index["source_url"].str.contains("kaggle", case=False).any(), (
        "В источниках внешних данных осталась ссылка Kaggle."
    )

    for _, row in index.iterrows():
        features_path = PROJECT_ROOT / row["features_file"]
        diagnostics_path = PROJECT_ROOT / row["diagnostics_file"]
        metadata_path = PROJECT_ROOT / row["metadata_file"]
        assert features_path.exists(), f"Не найден feature-CSV: {features_path}"
        assert diagnostics_path.exists(), f"Не найден diagnostics-CSV: {diagnostics_path}"
        assert metadata_path.exists(), f"Не найден metadata-файл: {metadata_path}"
        features = pd.read_csv(features_path)
        diagnostics = pd.read_csv(diagnostics_path)
        assert len(features) >= 1_000, f"Слишком малая внешняя выборка: {features_path.name}"
        assert len(features) == len(diagnostics), (
            f"Размеры feature-CSV и diagnostics-CSV различаются: {features_path.name}"
        )
        assert features["sample_id"].is_unique, f"sample_id не уникален: {features_path.name}"
        assert diagnostics["sample_id"].is_unique, f"sample_id не уникален: {diagnostics_path.name}"
        assert "is_allowed" in features.columns, f"Нет целевой классификационной переменной: {features_path.name}"
        assert features["is_allowed"].nunique() == 2, (
            f"Во внешней классификационной цели нет двух классов: {features_path.name}"
        )
        duplicated_after_merge = (set(features.columns) & set(diagnostics.columns)) - {"sample_id"}
        assert not duplicated_after_merge, (
            "Feature-CSV и diagnostics-CSV имеют дублирующиеся столбцы, "
            f"которые будут переименованы при merge: {sorted(duplicated_after_merge)}"
        )

    raw_motor_dir = PROJECT_ROOT / "data" / "raw" / "zenodo_motor_temperature"
    assert (raw_motor_dir / "ElectricMotorTemperature_TRAIN.ts").exists(), (
        "Не найден исходный TRAIN-файл Zenodo ElectricMotorTemperature."
    )
    assert (raw_motor_dir / "ElectricMotorTemperature_TEST.ts").exists(), (
        "Не найден исходный TEST-файл Zenodo ElectricMotorTemperature."
    )


def _check_external_notebooks() -> None:
    student_paths = sorted(NOTEBOOKS_EXTERNAL_STUDENT.glob("*.ipynb"))
    teacher_paths = sorted(NOTEBOOKS_EXTERNAL_TEACHER.glob("*.ipynb"))
    assert len(student_paths) == 9, f"Ожидалось 9 внешних студенческих блокнотов, найдено {len(student_paths)}."
    assert len(teacher_paths) == 9, f"Ожидалось 9 внешних преподавательских блокнотов, найдено {len(teacher_paths)}."
    assert not any("kaggle" in path.name.lower() for path in student_paths + teacher_paths), (
        "Среди внешних блокнотов остались устаревшие файлы Kaggle."
    )

    for path in student_paths:
        notebook = nbformat.read(path, as_version=4)
        notebook_source = "\n".join(str(cell.get("source", "")) for cell in notebook.cells)
        if path.name.startswith(("02_", "03_")):
            assert "forbidden_" in notebook_source and "Обнаружена утечка данных" in notebook_source, (
                f"Во внешнем студенческом блокноте нет программной защиты от утечки: {path.name}"
            )
        if path.name.startswith("03_"):
            assert "depth_metrics_df" in notebook_source, (
                f"Во внешнем блокноте занятия 3 нет анализа качества по max_depth: {path.name}"
            )
            assert "error_summary" in notebook_source, (
                f"Во внешнем блокноте занятия 3 нет анализа ошибок по диагностическим меткам: {path.name}"
            )
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert not cell.get("outputs"), f"Во внешнем студенческом блокноте есть output: {path.name}, cell {index}"
            assert cell.get("execution_count") is None, (
                f"Во внешнем студенческом блокноте есть execution_count: {path.name}, cell {index}"
            )

    for path in teacher_paths:
        notebook = nbformat.read(path, as_version=4)
        output_count = sum(len(cell.get("outputs", [])) for cell in notebook.cells if cell.cell_type == "code")
        assert output_count > 0, f"Преподавательский внешний блокнот не выполнен: {path.name}"


def main() -> None:
    _check_practice_01()
    _check_practice_02()
    _check_practice_03()
    _check_student_notebooks_clear_outputs()
    _check_dataset_catalog_status()
    _check_external_datasets()
    _check_external_notebooks()
    print("Инварианты курса для занятий 1-3 соблюдены.")


if __name__ == "__main__":
    main()

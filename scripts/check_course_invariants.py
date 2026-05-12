"""Smoke-test инвариантов курса для занятий 1-3.

Скрипт выполняет быстрые проверки, которые защищают учебные материалы от
методических регрессий: утечки целевой переменной, нарушения энергетического
баланса, случайного попадания диагностических столбцов в feature-CSV и
дисбаланса причин нарушения в классификационной задаче.
"""

from __future__ import annotations

from pathlib import Path
import re

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
EXTERNAL_FORBIDDEN_FEATURE_COLUMNS = {
    "zenodo_motor_temperature": {"target_temperature_c", "thermal_limit_target_temperature_c"},
    "zenodo_pmsm_inverter_fault": {
        "max_bridge_temp_c",
        "fault_code",
        "fault_label",
        "Current_Imbalance",
        "Temp_Diff_Max",
    },
    "mendeley_ev_powertrain_efficiency": {
        "motor_efficiency",
        "drivetrain_efficiency",
        "Powertrain_efficiency_gear_SG",
        "mechanical_power_w",
        "efficiency_limit",
        "class_label",
    },
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
    clean_df["temperature_c"] = clean_df["temperature_c"].fillna(clean_df["temperature_c"].median())
    omega_rad_s = 2.0 * np.pi * clean_df["speed_rpm"] / 60.0
    median_torque = clean_df["torque_nm"].median()
    median_current = clean_df["current_a"].median()
    target_max_efficiency = 0.98

    torque_missing_mask = clean_df["torque_nm"].isna()
    if torque_missing_mask.any():
        torque_cap = (
            target_max_efficiency
            * clean_df.loc[torque_missing_mask, "voltage_v"]
            * clean_df.loc[torque_missing_mask, "current_a"].fillna(median_current)
            / omega_rad_s.loc[torque_missing_mask]
        )
        clean_df.loc[torque_missing_mask, "torque_nm"] = np.minimum(median_torque, torque_cap)

    current_missing_mask = clean_df["current_a"].isna()
    if current_missing_mask.any():
        output_power = clean_df.loc[current_missing_mask, "torque_nm"] * omega_rad_s.loc[current_missing_mask]
        current_floor = output_power / (
            target_max_efficiency * clean_df.loc[current_missing_mask, "voltage_v"]
        )
        clean_df.loc[current_missing_mask, "current_a"] = np.maximum(median_current, current_floor)

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
    assert clean_df["efficiency"].between(0, 1).all(), "После очистки занятия 1 КПД вышел за [0, 1]."


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
        dataset_id = row["dataset_id"]
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
        forbidden = EXTERNAL_FORBIDDEN_FEATURE_COLUMNS.get(dataset_id, set())
        forbidden_in_features = forbidden.intersection(features.columns)
        assert not forbidden_in_features, (
            f"В feature-CSV внешнего набора попали запрещенные столбцы {features_path.name}: "
            f"{sorted(forbidden_in_features)}"
        )
        numeric_columns = [
            column
            for column in features.select_dtypes(include=[np.number]).columns
            if column != "sample_id"
        ]
        constant_columns = [
            column
            for column in numeric_columns
            if features[column].nunique(dropna=False) <= 1
        ]
        assert not constant_columns, (
            f"В feature-CSV найдены константные числовые столбцы {features_path.name}: "
            f"{constant_columns}"
        )
        if "profile_id" in features.columns:
            train_idx, test_idx = _external_group_holdout_indices(
                features,
                group_column="profile_id",
                target_column="is_allowed",
                test_share=0.25,
            )
            assert features.loc[train_idx, "is_allowed"].nunique() == 2, (
                f"Групповая train-выборка содержит не оба класса: {features_path.name}"
            )
            assert features.loc[test_idx, "is_allowed"].nunique() == 2, (
                f"Групповая test-выборка содержит не оба класса: {features_path.name}"
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


def _external_group_holdout_indices(
    data: pd.DataFrame,
    group_column: str,
    target_column: str,
    test_share: float = 0.25,
) -> tuple[pd.Index, pd.Index]:
    groups = np.array(sorted(data[group_column].dropna().unique()))
    test_count = max(1, int(np.ceil(len(groups) * test_share)))
    test_groups = groups[-test_count:]
    test_mask = data[group_column].isin(test_groups)
    train_idx = data.index[~test_mask]
    test_idx = data.index[test_mask]
    if (
        data.loc[train_idx, target_column].nunique() >= 2
        and data.loc[test_idx, target_column].nunique() >= 2
    ):
        return train_idx, test_idx

    selected_groups = []
    for class_value in sorted(data[target_column].dropna().unique()):
        class_groups = np.array(sorted(data.loc[data[target_column] == class_value, group_column].dropna().unique()))
        class_test_count = max(1, int(np.ceil(len(class_groups) * test_share)))
        selected_groups.extend(class_groups[-class_test_count:].tolist())
    test_groups = np.array(sorted(set(selected_groups)))
    test_mask = data[group_column].isin(test_groups)
    return data.index[~test_mask], data.index[test_mask]


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
        if path.name.startswith("01_"):
            assert "compact_numeric_profile" in notebook_source and "plot_missingness" in notebook_source, (
                f"Во внешнем блокноте занятия 1 нет расширенного аудита данных: {path.name}"
            )
        if path.name.startswith(("02_", "03_")):
            assert "forbidden_" in notebook_source and "Обнаружена утечка данных" in notebook_source, (
                f"Во внешнем студенческом блокноте нет программной защиты от утечки: {path.name}"
            )
        if path.name.startswith("02_"):
            assert "split_comparison" in notebook_source and "grid_metrics" in notebook_source, (
                f"Во внешнем блокноте занятия 2 нет сравнения разбиений или сетки экспериментов: {path.name}"
            )
        if path.name.startswith("03_"):
            assert "depth_metrics_df" in notebook_source, (
                f"Во внешнем блокноте занятия 3 нет анализа качества по max_depth: {path.name}"
            )
            assert "error_summary" in notebook_source, (
                f"Во внешнем блокноте занятия 3 нет анализа ошибок по диагностическим меткам: {path.name}"
            )
            assert "threshold_metrics" in notebook_source and "split_metrics" in notebook_source, (
                f"Во внешнем блокноте занятия 3 нет анализа порога решения или сравнения разбиений: {path.name}"
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
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                text = "".join(output.get("text", [])) if "text" in output else ""
                assert not re.search(r"\{[a-zA-Z_][^{}\n]{0,80}\}", text), (
                    f"В выводе преподавательского блокнота найден невычисленный шаблон: "
                    f"{path.name}, cell {cell_index}"
                )


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

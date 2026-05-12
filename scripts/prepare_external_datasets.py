"""Загрузка и подготовка компактных выборок внешних наборов данных.

Скрипт создает обработанные CSV для дополнительных блокнотов занятий 1-3.
Полные исходные архивы сохраняются в ``data/raw``. Компактные учебные
таблицы сохраняются в ``data/processed/external`` и имеют пару файлов:
``*_features.csv`` и ``*_diagnostics.csv``.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "external"
RANDOM_STATE = 20260507


@dataclass(frozen=True)
class ExternalDataset:
    dataset_id: str
    title_ru: str
    source_url: str
    license_note: str
    raw_dir: Path
    features_file: Path
    diagnostics_file: Path
    metadata_file: Path


DATASETS = {
    "zenodo_motor_temperature": ExternalDataset(
        dataset_id="zenodo_motor_temperature",
        title_ru="ElectricMotorTemperature, Zenodo TSML Archive",
        source_url="https://zenodo.org/records/11235562",
        license_note="CC BY 4.0; прямая загрузка файлов формата .ts из Zenodo",
        raw_dir=RAW_DIR / "zenodo_motor_temperature",
        features_file=PROCESSED_DIR / "zenodo_motor_temperature_features.csv",
        diagnostics_file=PROCESSED_DIR / "zenodo_motor_temperature_diagnostics.csv",
        metadata_file=PROCESSED_DIR / "zenodo_motor_temperature_metadata.md",
    ),
    "zenodo_pmsm_inverter_fault": ExternalDataset(
        dataset_id="zenodo_pmsm_inverter_fault",
        title_ru="Zenodo PMSM inverter fault diagnosis",
        source_url="https://zenodo.org/records/14482932",
        license_note="CC BY 4.0",
        raw_dir=RAW_DIR / "zenodo_pmsm_inverter_fault",
        features_file=PROCESSED_DIR / "zenodo_pmsm_inverter_fault_features.csv",
        diagnostics_file=PROCESSED_DIR / "zenodo_pmsm_inverter_fault_diagnostics.csv",
        metadata_file=PROCESSED_DIR / "zenodo_pmsm_inverter_fault_metadata.md",
    ),
    "mendeley_ev_powertrain_efficiency": ExternalDataset(
        dataset_id="mendeley_ev_powertrain_efficiency",
        title_ru="Processed Data for EV Powertrain Efficiency, Mendeley Data",
        source_url="https://data.mendeley.com/datasets/kbwr2z8r3y",
        license_note="CC BY 4.0",
        raw_dir=RAW_DIR / "mendeley_ev_powertrain_efficiency",
        features_file=PROCESSED_DIR / "mendeley_ev_powertrain_efficiency_features.csv",
        diagnostics_file=PROCESSED_DIR / "mendeley_ev_powertrain_efficiency_diagnostics.csv",
        metadata_file=PROCESSED_DIR / "mendeley_ev_powertrain_efficiency_metadata.md",
    ),
}


def _safe_mkdir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS.values():
        dataset.raw_dir.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)


def _extract_zip(archive: Path, target_dir: Path) -> None:
    marker = target_dir / ".extracted"
    if marker.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(target_dir)
    marker.write_text("ok\n", encoding="utf-8")


def download_external_sources() -> None:
    """Скачать открытые архивы из источников с прямым доступом."""

    _safe_mkdir()

    motor_temperature = DATASETS["zenodo_motor_temperature"]
    _download_file(
        "https://zenodo.org/api/records/11235562/files/ElectricMotorTemperature_TRAIN.ts/content",
        motor_temperature.raw_dir / "ElectricMotorTemperature_TRAIN.ts",
    )
    _download_file(
        "https://zenodo.org/api/records/11235562/files/ElectricMotorTemperature_TEST.ts/content",
        motor_temperature.raw_dir / "ElectricMotorTemperature_TEST.ts",
    )

    zenodo_zip = DATASETS["zenodo_pmsm_inverter_fault"].raw_dir / "PMSM-inverter-fault-diagnosis-3.0.zip"
    _download_file(
        "https://zenodo.org/api/records/14482932/files/PMSM-inverter-fault-diagnosis-3.0.zip/content",
        zenodo_zip,
    )
    _extract_zip(zenodo_zip, DATASETS["zenodo_pmsm_inverter_fault"].raw_dir / "extracted")

    mendeley_zip = DATASETS["mendeley_ev_powertrain_efficiency"].raw_dir / "kbwr2z8r3y-1.zip"
    _download_file(
        "https://data.mendeley.com/public-api/zip/kbwr2z8r3y/download/1",
        mendeley_zip,
    )
    _extract_zip(mendeley_zip, DATASETS["mendeley_ev_powertrain_efficiency"].raw_dir / "extracted")


def _compact_by_profile(df: pd.DataFrame, profile_column: str, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    groups = list(df.groupby(profile_column, sort=True))
    per_group = max(1, max_rows // max(1, len(groups)))
    for _, group in groups:
        if len(group) <= per_group:
            parts.append(group)
        else:
            parts.append(group.sample(n=per_group, random_state=int(rng.integers(0, 1_000_000))))
    result = pd.concat(parts, axis=0).sort_index()
    if len(result) > max_rows:
        result = result.sample(n=max_rows, random_state=RANDOM_STATE).sort_index()
    return result.reset_index(drop=True)


def _write_metadata(dataset: ExternalDataset, status: str, rows: int, extra: list[str] | None = None) -> None:
    extra = extra or []
    text = [
        f"# {dataset.title_ru}",
        "",
        f"- Идентификатор: `{dataset.dataset_id}`.",
        f"- Источник: {dataset.source_url}.",
        f"- Лицензия и доступ: {dataset.license_note}.",
        f"- Статус подготовки: `{status}`.",
        f"- Число строк в компактной учебной выборке: {rows}.",
        "",
        "Методическое назначение: расширенные блокноты занятий 1-3 с реальными",
        "или открытыми инженерными данными. Feature-CSV используется для базовой",
        "модели, diagnostics-CSV - для пояснения происхождения целевых переменных,",
        "ограничений и возможной утечки данных.",
    ]
    if extra:
        text.extend(["", "Дополнительные замечания:"])
        text.extend(f"- {item}" for item in extra)
    dataset.metadata_file.write_text("\n".join(text) + "\n", encoding="utf-8")


def _read_ts_archive(path: Path, split_name: str, max_rows: int) -> pd.DataFrame:
    """Прочитать ограниченное число строк из multivariate .ts-файла TSML.

    Формат .ts хранит один временной фрагмент в строке. Каналы разделяются
    двоеточием, значения внутри канала разделяются запятыми, последний элемент
    строки является целевой переменной регрессии. Архив Zenodo не содержит
    исходных физических имен каналов, поэтому в учебных таблицах используются
    нейтральные имена `channel_XX`.
    """

    records: list[dict[str, float | int | str]] = []
    data_started = False
    with path.open("r", encoding="utf-8") as file_obj:
        for source_row_index, line in enumerate(file_obj):
            line = line.strip()
            if not line:
                continue
            if line.lower() == "@data":
                data_started = True
                continue
            if not data_started or line.startswith("@"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            channels = [np.fromstring(part, sep=",", dtype=float) for part in parts[:-1]]
            target_temperature_c = float(parts[-1])
            record: dict[str, float | int | str] = {
                "source_split": split_name,
                "source_row_index": source_row_index,
                "target_temperature_c": target_temperature_c,
            }
            for channel_index, values in enumerate(channels):
                if values.size == 0:
                    continue
                prefix = f"channel_{channel_index:02d}"
                record[f"{prefix}_mean"] = float(values.mean())
                record[f"{prefix}_std"] = float(values.std(ddof=0))
                record[f"{prefix}_min"] = float(values.min())
                record[f"{prefix}_max"] = float(values.max())
                record[f"{prefix}_first"] = float(values[0])
                record[f"{prefix}_last"] = float(values[-1])
                record[f"{prefix}_trend"] = float(values[-1] - values[0])
            records.append(record)
            if len(records) >= max_rows:
                break
    return pd.DataFrame.from_records(records)


def prepare_zenodo_motor_temperature() -> bool:
    dataset = DATASETS["zenodo_motor_temperature"]
    train_source = dataset.raw_dir / "ElectricMotorTemperature_TRAIN.ts"
    test_source = dataset.raw_dir / "ElectricMotorTemperature_TEST.ts"
    if not train_source.exists() or not test_source.exists():
        return False

    train = _read_ts_archive(train_source, "train", max_rows=8_000)
    test = _read_ts_archive(test_source, "test", max_rows=4_000)
    raw = pd.concat([train, test], ignore_index=True)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    raw["sample_id"] = np.arange(1, len(raw) + 1)
    raw["profile_id"] = (
        raw["source_split"].map({"train": 1_000, "test": 2_000}).astype(int)
        + raw.groupby("source_split").cumcount() // 200
    )
    raw["time_index"] = raw.groupby("profile_id").cumcount()
    temperature_limit = raw["target_temperature_c"].quantile(0.80)
    raw["is_allowed"] = (raw["target_temperature_c"] <= temperature_limit).astype(int)

    feature_columns = [
        "sample_id",
        "profile_id",
        "time_index",
        "channel_00_mean",
        "channel_01_mean",
        "channel_02_mean",
        "channel_03_mean",
        "channel_04_mean",
        "channel_05_mean",
        "channel_00_trend",
        "channel_01_trend",
        "is_allowed",
    ]
    diagnostics_columns = [
        "sample_id",
        "source_split",
        "source_row_index",
        "target_temperature_c",
        "channel_00_std",
        "channel_01_std",
        "channel_02_std",
        "channel_03_std",
        "channel_04_std",
        "channel_05_std",
        "channel_00_min",
        "channel_01_min",
        "channel_02_min",
        "channel_03_min",
        "channel_04_min",
        "channel_05_min",
        "channel_00_max",
        "channel_01_max",
        "channel_02_max",
        "channel_03_max",
        "channel_04_max",
        "channel_05_max",
    ]
    features = raw[feature_columns].copy()
    diagnostics = raw[diagnostics_columns].copy()
    diagnostics["thermal_limit_target_temperature_c"] = temperature_limit
    diagnostics["class_label"] = np.where(
        raw["is_allowed"] == 1,
        "normal_or_moderate_target_temperature",
        "high_target_temperature",
    )
    features.round(6).to_csv(dataset.features_file, index=False)
    diagnostics.round(6).to_csv(dataset.diagnostics_file, index=False)
    _write_metadata(
        dataset,
        status="prepared",
        rows=len(features),
        extra=[
            "Формат исходных файлов: multivariate .ts, то есть многомерные временные ряды.",
            "Архив содержит стандартизованные временные фрагменты длиной 60 отсчетов и целевую температуру.",
            "Целевая переменная регрессии: `target_temperature_c`.",
            "Целевая переменная классификации: `is_allowed`, производная от квантиля целевой температуры.",
            "Физические имена каналов в .ts-файлах не заданы, поэтому признаки названы нейтрально: `channel_00_mean`, `channel_01_mean` и далее.",
            "Для проверки качества используется групповое разбиение по `profile_id`, сформированному из исходного split и блока строк архива.",
        ],
    )
    return True


def prepare_zenodo_pmsm_inverter_fault() -> bool:
    dataset = DATASETS["zenodo_pmsm_inverter_fault"]
    source = dataset.raw_dir / "extracted" / "PMSM-inverter-fault-diagnosis-3.0" / "processed_data" / "converted_dataset.csv"
    if not source.exists():
        return False

    raw = pd.read_csv(source)
    raw = raw.dropna(subset=["Ia_original", "Ib_original", "VDC", "IDC", "T1", "T2", "T3", "FDD"]).copy()
    raw["fault_group_id"] = raw["FDD"].astype("category").cat.codes + 1
    df = _compact_by_profile(raw, "fault_group_id", max_rows=12_000)
    df = df.reset_index(drop=True)
    df["profile_id"] = (
        df["fault_group_id"].astype(int) * 1_000
        + df.groupby("fault_group_id").cumcount() // 250
        + 1
    )
    sample_id = np.arange(1, len(df) + 1)
    omega_rad_s = 10.0
    current_a = np.sqrt(df["Ia_original"] ** 2 + df["Ib_original"] ** 2)
    voltage_v = df["VDC"].abs()
    temperature_c = df[["T1", "T2", "T3"]].mean(axis=1)
    torque_proxy_nm = (df["Power_AC"].abs() / omega_rad_s).clip(upper=df["Power_AC"].abs().quantile(0.99))
    is_allowed = (df["FDD"].astype(str) == "F0").astype(int)

    features = pd.DataFrame(
        {
            "sample_id": sample_id,
            "profile_id": df["profile_id"].astype(int),
            "time_index": df["Timestamp"],
            "torque_proxy_nm": torque_proxy_nm,
            "voltage_v": voltage_v,
            "current_a": current_a,
            "temperature_c": temperature_c,
            "is_allowed": is_allowed,
        }
    )
    diagnostics = pd.DataFrame(
        {
            "sample_id": sample_id,
            "max_bridge_temp_c": df[["T1", "T2", "T3"]].max(axis=1),
            "fault_code": df["FDD"].astype(str),
            "fault_label": np.where(is_allowed == 1, "normal_operation", "fault_condition"),
            "T1": df["T1"],
            "T2": df["T2"],
            "T3": df["T3"],
            "Ia_original": df["Ia_original"],
            "Ib_original": df["Ib_original"],
            "IDC_original": df["IDC_original"],
            "Power_DC": df["Power_DC"],
            "Power_AC": df["Power_AC"],
            "Current_Imbalance": df["Current_Imbalance"],
            "Temp_Diff_Max": df["Temp_Diff_Max"],
        }
    )
    features.round(6).to_csv(dataset.features_file, index=False)
    diagnostics.round(6).to_csv(dataset.diagnostics_file, index=False)
    _write_metadata(
        dataset,
        status="prepared",
        rows=len(features),
        extra=[
            "Целевая переменная регрессии: `max_bridge_temp_c`.",
            "Целевая переменная классификации: `is_allowed`, где `F0` трактуется как нормальная работа.",
            "`torque_proxy_nm` является расчетным прокси-показателем, полученным из `Power_AC` и фиксированной угловой скорости 10 рад/с; сама скорость не включена в feature-CSV, так как в источнике она не изменяется.",
            "`profile_id` сформирован как сегмент внутри кода отказа, чтобы групповое тестирование включало нормальные и отказные режимы.",
        ],
    )
    return True


def prepare_mendeley_ev_powertrain_efficiency() -> bool:
    dataset = DATASETS["mendeley_ev_powertrain_efficiency"]
    root = dataset.raw_dir / "extracted" / "Processed Data for Electric Vehicle Powertrain Efficiency Integrating Driving Cycle Data and Electric Motor Efficiency Maps"
    source = root / "Tracking_data_efficiecny.csv"
    if not source.exists():
        return False

    raw = pd.read_csv(source)
    raw = raw.dropna(
        subset=[
            "Date",
            "Velocity",
            "Acceleration",
            "Slope Angle (rad)",
            "Motor_rpm_gear_SG",
            "Motor_torque_gear_SG",
            "Motor_efficiency_gear_SG",
            "Drivetrain_efficiency_gear_SG",
        ]
    ).copy()
    raw["motor_efficiency_clipped"] = raw["Motor_efficiency_gear_SG"].clip(lower=0.0, upper=1.0)
    non_plateau = raw[raw["motor_efficiency_clipped"] < 0.995]
    plateau = raw[raw["motor_efficiency_clipped"] >= 0.995]
    rng = np.random.default_rng(RANDOM_STATE)
    non_plateau_sample = non_plateau.sample(
        n=min(len(non_plateau), 7_000),
        random_state=int(rng.integers(0, 1_000_000)),
    )
    plateau_sample = plateau.sample(
        n=min(len(plateau), 5_000),
        random_state=int(rng.integers(0, 1_000_000)),
    )
    df = (
        pd.concat([non_plateau_sample, plateau_sample], axis=0)
        .sort_values(["Date", "DateTime"])
        .reset_index(drop=True)
    )
    date_code = (df["Date"].astype("category").cat.codes + 1).astype(int)
    segment_id = df.groupby("Date").cumcount() // 500
    df["profile_id"] = (date_code * 1_000 + segment_id + 1).astype(int)
    sample_id = np.arange(1, len(df) + 1)
    motor_efficiency = df["motor_efficiency_clipped"]
    drivetrain_efficiency = df["Drivetrain_efficiency_gear_SG"].clip(lower=0.0, upper=1.0)
    efficiency_limit = motor_efficiency.quantile(0.25)
    omega = 2.0 * np.pi * df["Motor_rpm_gear_SG"].abs() / 60.0
    mechanical_power_w = df["Motor_torque_gear_SG"].abs() * omega

    features = pd.DataFrame(
        {
            "sample_id": sample_id,
            "profile_id": df["profile_id"].astype(int),
            "time_index": pd.to_datetime(df["DateTime"], errors="coerce").astype("int64") // 10**9,
            "vehicle_speed_m_s": df["Velocity"],
            "acceleration_m_s2": df["Acceleration"],
            "slope_rad": df["Slope Angle (rad)"],
            "motor_speed_rpm": df["Motor_rpm_gear_SG"],
            "motor_torque_nm": df["Motor_torque_gear_SG"],
            "is_allowed": (motor_efficiency >= efficiency_limit).astype(int),
        }
    )
    diagnostics = pd.DataFrame(
        {
            "sample_id": sample_id,
            "motor_efficiency": motor_efficiency,
            "drivetrain_efficiency": drivetrain_efficiency,
            "Date": df["Date"],
            "DateTime": df["DateTime"],
            "Latitude": df["Latitude"],
            "Longitude": df["Longitude"],
            "Total Resistive Force": df["Total Resistive Force"],
            "mechanical_power_w": mechanical_power_w,
            "Powertrain_efficiency_gear_SG": df["Powertrain_efficiency_gear_SG"],
            "efficiency_limit": efficiency_limit,
            "class_label": np.where(motor_efficiency >= efficiency_limit, "acceptable_efficiency", "low_efficiency"),
        }
    )
    features.round(6).to_csv(dataset.features_file, index=False)
    diagnostics.round(6).to_csv(dataset.diagnostics_file, index=False)
    _write_metadata(
        dataset,
        status="prepared",
        rows=len(features),
        extra=[
            "Целевая переменная регрессии: `motor_efficiency`, вынесенная в diagnostics-CSV для защиты feature-CSV от прокси-утечки.",
            "Целевая переменная классификации: `is_allowed`, производная от нижнего квартиля `motor_efficiency`.",
            "Компактная выборка намеренно переобогащена точками с `motor_efficiency < 0.995`, чтобы регрессионная задача не вырождалась в прогноз почти постоянной величины.",
            "Для проверки качества используется разбиение по временным сегментам поездок, а не случайное перемешивание соседних точек траектории.",
        ],
    )
    return True


def write_external_index(prepared: dict[str, bool]) -> None:
    rows = []
    for dataset_id, dataset in DATASETS.items():
        rows.append(
            {
                "dataset_id": dataset_id,
                "title_ru": dataset.title_ru,
                "source_url": dataset.source_url,
                "license_note": dataset.license_note,
                "features_file": dataset.features_file.relative_to(PROJECT_ROOT).as_posix(),
                "diagnostics_file": dataset.diagnostics_file.relative_to(PROJECT_ROOT).as_posix(),
                "metadata_file": dataset.metadata_file.relative_to(PROJECT_ROOT).as_posix(),
                "prepared": bool(prepared.get(dataset_id, False)),
            }
        )
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "external_dataset_index.csv", index=False)
    (PROCESSED_DIR / "external_dataset_index.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    _safe_mkdir()
    download_external_sources()
    prepared = {
        "zenodo_motor_temperature": prepare_zenodo_motor_temperature(),
        "zenodo_pmsm_inverter_fault": prepare_zenodo_pmsm_inverter_fault(),
        "mendeley_ev_powertrain_efficiency": prepare_mendeley_ev_powertrain_efficiency(),
    }
    write_external_index(prepared)
    for dataset_id, status in prepared.items():
        print(f"{dataset_id}: {'prepared' if status else 'not prepared'}")


if __name__ == "__main__":
    main()

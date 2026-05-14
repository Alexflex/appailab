"""Предварительная загрузка внешних данных для преподавателя.

Скрипт предназначен для подготовки аудитории до занятия. Он скачивает
открытые внешние архивы, проверяет контрольные суммы SHA256 и формирует
компактные CSV в ``data/processed/external``.
"""

from __future__ import annotations

from prepare_external_datasets import (
    download_external_sources,
    prepare_mendeley_ev_powertrain_efficiency,
    prepare_zenodo_motor_temperature,
    prepare_zenodo_pmsm_inverter_fault,
    write_external_index,
)


def main() -> None:
    print("Предварительная загрузка внешних данных курса.")
    print("Этап 1: загрузка raw-архивов и проверка SHA256.")
    download_external_sources()

    print("Этап 2: подготовка компактных учебных CSV.")
    prepared = {
        "zenodo_motor_temperature": prepare_zenodo_motor_temperature(),
        "zenodo_pmsm_inverter_fault": prepare_zenodo_pmsm_inverter_fault(),
        "mendeley_ev_powertrain_efficiency": prepare_mendeley_ev_powertrain_efficiency(),
    }
    write_external_index(prepared)

    for dataset_id, status in prepared.items():
        print(f"{dataset_id}: {'prepared' if status else 'not prepared'}")

    if not all(prepared.values()):
        raise RuntimeError(
            "Не все внешние наборы данных подготовлены. Проверьте сообщения выше "
            "и доступность исходных архивов."
        )

    print("Внешние данные подготовлены.")


if __name__ == "__main__":
    main()

# Учебные наборы данных для практических занятий 1-3

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

# Учебные наборы данных для практических занятий 1-9

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
   разрядов: амплитуда, RMS, энергия, число импульсов, спектральные
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

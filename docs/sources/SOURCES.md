# Проверенные источники для занятий 1-3

Дата проверки актуальности: 2026-05-07.

Документ содержит основной проверенный перечень источников для базовых и
расширенных заданий. Полный каталог из 25 наборов данных приведен в
`data/processed/practice_01_03_dataset_catalog.csv`, а подробное описание
всех 46 заданий - в `docs/sources/dataset_assignment_details.md`.

## Нормативные и методические источники

1. IEC 60034-1:2026. Rotating electrical machines - Part 1: Rating and performance. URL: https://webstore.iec.ch/en/publication/89961.
2. IEC 60034-2-1:2024. Rotating electrical machines - Part 2-1: Standard methods for determining losses and efficiency from tests. URL: https://webstore.iec.ch/en/publication/67756.
3. Документация Jupyter. URL: https://docs.jupyter.org/.
4. Документация pandas по пропущенным данным. URL: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html.
5. Документация scikit-learn по гребневой регрессии `Ridge`, классификатору дерева решений `DecisionTreeClassifier`, метрикам и разбиению выборок. URL: https://scikit-learn.org/stable/.

## Основные наборы данных для занятий 1-2

1. Electric Motor Temperature, Kaggle. URL: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature.
   Лицензия: CC BY-SA 4.0. Реальные стендовые измерения постоянно-магнитной синхронной машины (Permanent Magnet Synchronous Motor, PMSM). Рекомендуется как основной реальный источник для анализа признаков и регрессии температур.

2. Comprehensive Dataset for Fault Detection and Diagnosis in Inverter-Driven PMSM Systems, Zenodo. URL: https://zenodo.org/records/14482932.
   Лицензия: CC BY 4.0. Компактный набор по PMSM-инвертору с токами, напряжениями, температурами и режимами отказов. Рекомендуется как расширение для классификации и регрессии температурных признаков.

3. Processed Data for EV Powertrain Efficiency, Mendeley Data. URL: https://data.mendeley.com/datasets/kbwr2z8r3y.
   Лицензия: CC BY 4.0. Набор для анализа эффективности электропривода транспортного средства. Рекомендуется как дополнительный источник по энергетическим характеристикам, но не как основной стендовый набор двигателя.

4. Identifying the Physics Behind an Electric Motor, Kaggle. URL: https://www.kaggle.com/datasets/hankelea/system-identification-of-an-electric-motor.
   Доступ публичный через Kaggle, но условия переиздания данных менее ясны. Рекомендуется только как продвинутый дополнительный источник.

5. Permanent Magnet DC Worm Geared Motor Data, Mendeley Data. URL: https://data.mendeley.com/datasets/2rkpsss6fd/2.
   Лицензия: CC BY 4.0. Набор лабораторных измерений двигателя постоянного тока с постоянными магнитами и червячным редуктором. Рекомендуется как понятный источник для первичного анализа и регрессии скорости или тока.

## Основные наборы данных для занятия 3

1. Electric Motor Temperature, Kaggle. URL: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature.
   Может использоваться для построения целевой переменной "допустимый тепловой режим" на основе порога температуры статора или постоянных магнитов. Требуется явно обосновать порог и исключить утечку данных.

2. UCI AI4I 2020 Predictive Maintenance Dataset. URL: https://archive.ics.uci.edu/dataset/601/ai4i.
   Лицензия: CC BY 4.0. Малый дидактический набор для объяснения дерева решений и метрик классификации. Ограничение: синтетический характер и неполное соответствие тематике электропривода.

3. Paderborn University Bearing Data Center. URL: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter.
   Лицензия: CC BY-NC 4.0. Реальные данные электромеханического стенда: токи двигателя, вибрации, скорость, момент, радиальная нагрузка и температура. Рекомендуется для расширенного занятия или проекта.

4. Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis, Mendeley Data. URL: https://data.mendeley.com/datasets/ztmf3m7h5x/6.
   Лицензия: CC BY 4.0. Реальные данные вращающейся машины при нормальном состоянии, дефектах подшипников, несоосности и дисбалансе ротора. Требуется извлечение признаков из сигналов.

5. MOTOR FAULT DETECTION DATA, Figshare. URL: https://figshare.com/articles/dataset/MOTOR_FAULT_DETECTION_DATA/27216219.
   Лицензия: CC BY 4.0. Данные трехфазного асинхронного двигателя с синхронными сигналами вибрации, тока и напряжения. Из-за объема около 5 ГБ для аудиторного занятия требуется предварительно подготовленное подмножество.

6. Case Western Reserve University Bearing Data Center. URL: https://engineering.case.edu/bearingdatacenter/welcome.
   Публичный эталонный сравнительный набор данных (benchmark dataset) по вибрационной диагностике подшипников электродвигателя. Явная лицензия на официальной странице не указана, поэтому для учебного комплекта предпочтительнее использовать ссылку и малые производные признаки только при соблюдении условий цитирования.

"""Сборка базовых Jupyter Notebook для практических занятий 7-9."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import v4 as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_STUDENT = PROJECT_ROOT / "notebooks" / "student"
NOTEBOOKS_TEACHER = PROJECT_ROOT / "notebooks" / "teacher"


def md(source: str) -> nbformat.NotebookNode:
    return nbf.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbf.new_code_cell(dedent(source).strip())


def save_notebook(cells: list[nbformat.NotebookNode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbformat.write(nb, path)


def colab_bootstrap_cells(required_processed_files: list[str]) -> list[nbformat.NotebookNode]:
    required_repr = repr(required_processed_files)
    return [
        md(
            """
            ## Инициализация среды выполнения

            Ячейка ниже обеспечивает запуск блокнота в Google Colab и в
            локальном Jupyter Notebook. Если проект уже открыт локально,
            повторное клонирование не выполняется.
            """
        ),
        code(
            f"""
            # COLAB_BOOTSTRAP_APPailab
            from pathlib import Path
            import os
            import sys

            REQUIRED_PROCESSED_FILES = {required_repr}
            PROJECT_REPOSITORY_URL = "https://github.com/Alexflex/appailab.git"

            def find_project_root(start: Path) -> Path | None:
                for candidate in [start, *start.parents]:
                    if (candidate / "requirements-colab.txt").exists() and (candidate / "data" / "processed").exists():
                        return candidate
                return None

            project_root = find_project_root(Path.cwd())

            if project_root is None:
                try:
                    import google.colab  # type: ignore
                    IN_COLAB = True
                except Exception:
                    IN_COLAB = False

                if IN_COLAB:
                    workdir = Path("/content/appailab")
                    if not workdir.exists():
                        !git clone -q {{PROJECT_REPOSITORY_URL}} {{workdir}}
                    project_root = workdir
                    os.chdir(project_root)
                    !pip install -q -r requirements-colab.txt
                else:
                    raise FileNotFoundError(
                        "Не найден корень проекта. Откройте блокнот из репозитория appailab "
                        "или выполните git clone перед запуском."
                    )

            sys.path.insert(0, str(project_root / "src"))
            missing_files = [
                name for name in REQUIRED_PROCESSED_FILES
                if not (project_root / "data" / "processed" / name).exists()
            ]
            if missing_files:
                raise FileNotFoundError(
                    "Не найдены подготовленные CSV: " + ", ".join(missing_files)
                    + ". Выполните python scripts/generate_datasets.py."
                )

            print(f"Корень проекта: {{project_root}}")
            print("Проверенные CSV:", ", ".join(REQUIRED_PROCESSED_FILES))
            """
        ),
    ]


COMMON_IMPORTS = """
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11

DATA_DIR = project_root / "data" / "processed"
CATALOG_07_09_FILE = DATA_DIR / "practice_07_09_dataset_catalog.csv"
ASSIGNMENTS_07_09_FILE = DATA_DIR / "practice_07_09_dataset_assignments.csv"
RANDOM_STATE = 20260507
"""


def todo_list(teacher: bool, values: list[str], name: str, comment: str) -> str:
    if teacher:
        return f"{name} = {values!r}\n"
    return (
        f"# TODO: заполните список. {comment}\n"
        f"# Рекомендуемый первый вариант: {values!r}\n"
        f"{name} = None\n"
        f"if {name} is None:\n"
        f"    raise ValueError('Заполните {name}: {comment}')\n"
    )


def todo_value(teacher: bool, value: object, name: str, comment: str) -> str:
    if teacher:
        return f"{name} = {value!r}\n"
    return (
        f"# TODO: задайте значение. {comment}\n"
        f"# Рекомендуемое значение: {value!r}\n"
        f"{name} = None\n"
        f"if {name} is None:\n"
        f"    raise ValueError('Заполните {name}: {comment}')\n"
    )


def todo_text(teacher: bool, value: str, name: str, comment: str) -> str:
    if teacher:
        return f"{name} = {value!r}\nprint({name})\n"
    return (
        f"# TODO: впишите краткий вывод. {comment}\n"
        f"{name} = ''\n"
        f"if not {name}.strip():\n"
        f"    raise ValueError('Заполните {name}: {comment}')\n"
        f"print({name})\n"
    )


def source_section_07_09() -> list[nbformat.NotebookNode]:
    return [
        md(
            """
            ## Источники и проверка актуальности

            Базовые занятия 7-9 используют локальные воспроизводимые CSV. Открытые
            источники ниже применяются как методические задания: студент должен
            определить объект, признаки, целевую переменную, риски утечки данных
            и допустимые визуализации.
            """
        ),
        code(
            """
            catalog_07_09 = pd.read_csv(CATALOG_07_09_FILE)
            assignments_07_09 = pd.read_csv(ASSIGNMENTS_07_09_FILE)
            display(catalog_07_09[[
                "dataset_id", "name", "lessons", "implementation_status",
                "risk_level", "checked_at"
            ]])
            """
        ),
        md(
            """
            ## Реестр найденных наборов данных и развернутые задания

            Все источники имеют статус `methodology_only`, то есть на данном
            этапе они не являются обязательными для выполнения в аудитории.
            """
        ),
        code(
            """
            display(assignments_07_09[[
                "assignment_id", "lesson", "assignment_title",
                "recommended_visualizations", "control_questions"
            ]])
            """
        ),
    ]


def build_practice_07(teacher: bool) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = [
        md(
            f"""
            # Практическое занятие 7. Анализ синтетических сигналов частичных разрядов

            Версия: {"преподавательская" if teacher else "студенческая"}.

            Цель занятия - показать полный переход от временного сигнала к
            признакам и простой классификационной модели. Частичный разряд
            (partial discharge, PD) - локальный электрический пробой малой
            части изоляции, который не замыкает электроды полностью, но
            является важным диагностическим признаком старения изоляции.
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_07_pd_signal_features.csv",
                "practice_07_pd_signal_diagnostics.csv",
                "practice_07_pd_signal_waveforms.csv",
                "practice_07_09_dataset_catalog.csv",
                "practice_07_09_dataset_assignments.csv",
            ]
        ),
        code(COMMON_IMPORTS),
        md(
            """
            ## Теоретический блок

            Временная область описывает изменение напряжения `x(t)` во времени.
            Частотная область описывает, на каких частотах сосредоточена энергия
            сигнала. Быстрое преобразование Фурье (Fast Fourier Transform, FFT)
            является алгоритмом вычисления дискретного спектра.

            Энергия дискретного сигнала оценивается как

            $$
            E = \\sum_{i=1}^{N} x_i^2 \\Delta t,
            $$

            где `x_i` - отсчет сигнала, `Delta t` - шаг дискретизации. Отношение
            сигнал-шум (Signal-to-Noise Ratio, SNR) показывает, насколько
            полезный сигнал превышает уровень шума.
            """
        ),
        md(
            """
            ## Последовательность работы

            1. Загрузить таблицу признаков, diagnostics-CSV и временные отсчеты.
            2. Построить временные графики для нескольких классов состояния.
            3. Построить амплитудные спектры и сравнить доминирующие частоты.
            4. Выбрать безопасные признаки без прямой диагностической разметки.
            5. Обучить классификатор и проанализировать матрицу ошибок.
            6. Разобрать антипример утечки данных.
            """
        ),
        code(
            """
            features_df = pd.read_csv(DATA_DIR / "practice_07_pd_signal_features.csv")
            diagnostics_df = pd.read_csv(DATA_DIR / "practice_07_pd_signal_diagnostics.csv")
            waveforms_df = pd.read_csv(DATA_DIR / "practice_07_pd_signal_waveforms.csv")

            print("features:", features_df.shape)
            print("diagnostics:", diagnostics_df.shape)
            print("waveforms:", waveforms_df.shape)
            display(features_df.head())
            """
        ),
        md(
            """
            ## Паспорт данных

            Одна строка feature-CSV соответствует одному окну сигнала. Одна
            строка waveforms-CSV соответствует одному отсчету одного окна. Целевая
            переменная `condition_class` задает учебное состояние: нормальный
            сигнал, редкие разряды, частые разряды, шумовой режим или внешняя
            помеха.
            """
        ),
        code(
            """
            display(features_df.groupby("condition_class").agg(
                n=("sample_id", "count"),
                mean_energy=("signal_energy", "mean"),
                mean_snr=("snr_db", "mean"),
                mean_pulses=("pulse_count_est", "mean"),
            ).round(4))
            """
        ),
        md(
            """
            ## Временные сигналы

            На графиках ниже видно, что амплитуда сама по себе не всегда
            достаточна: шумовой режим может иметь заметную амплитуду, но не
            содержать устойчивой импульсной структуры.
            """
        ),
        code(
            """
            example_ids = (
                features_df.groupby("condition_class")["sample_id"]
                .first()
                .to_dict()
            )
            fig, axes = plt.subplots(len(example_ids), 1, figsize=(10, 11), sharex=True)
            for ax, (label, sample_id) in zip(axes, example_ids.items()):
                part = waveforms_df[waveforms_df["sample_id"] == sample_id]
                ax.plot(part["time_us"], part["voltage_v"], linewidth=1.2)
                ax.set_title(label)
                ax.set_ylabel("U, V")
            axes[-1].set_xlabel("Время, microseconds")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Спектральный анализ

            Спектр помогает отделить импульсные события от гармонической внешней
            помехи. Внешняя помеха может иметь выраженный частотный пик, тогда
            как импульсный разряд распределяет энергию шире.
            """
        ),
        code(
            """
            def plot_spectrum(sample_id: int, label: str) -> None:
                part = waveforms_df[waveforms_df["sample_id"] == sample_id].sort_values("point_index")
                signal = part["voltage_v"].to_numpy()
                dt = (part["time_us"].iloc[1] - part["time_us"].iloc[0]) * 1e-6
                freq_khz = np.fft.rfftfreq(len(signal), d=dt) / 1000.0
                spectrum = np.abs(np.fft.rfft(signal))
                plt.plot(freq_khz, spectrum, label=label)

            plt.figure(figsize=(10, 5))
            for label, sample_id in example_ids.items():
                plot_spectrum(sample_id, label)
            plt.xlim(0, 800)
            plt.xlabel("Частота, kHz")
            plt.ylabel("Амплитуда спектра")
            plt.title("Сравнение спектров учебных сигналов")
            plt.legend()
            plt.show()
            """
        ),
        md(
            """
            ## Выбор признаков

            Безопасные признаки - это величины, которые можно рассчитать из
            сигнала до знания целевого класса. Диагностические столбцы
            `state_code`, `direct_state_label` и `leakage_pd_indicator` не
            включаются в базовую модель.
            """
        ),
        code(
            todo_list(
                teacher,
                [
                    "max_abs_voltage_v",
                    "rms_voltage_v",
                    "signal_energy",
                    "pulse_count_est",
                    "dominant_freq_khz",
                    "spectral_centroid_khz",
                    "snr_db",
                ],
                "signal_features",
                "Выберите 5-7 признаков, рассчитанных из временного окна.",
            )
            + "\nforbidden_signal_features = {'state_code', 'direct_state_label', 'leakage_pd_indicator'}\n"
            + "leaked = forbidden_signal_features & set(signal_features)\n"
            + "if leaked:\n"
            + "    raise ValueError(f'Обнаружена утечка данных: {sorted(leaked)}')\n"
        ),
        code(
            todo_value(
                teacher,
                160,
                "max_iter",
                "Задайте число итераций LogisticRegression в диапазоне 100..400.",
            )
        ),
        md(
            """
            ## Базовая классификационная модель

            Логистическая регрессия (Logistic Regression) оценивает вероятность
            класса через линейную комбинацию признаков. Масштабирование
            необходимо, потому что энергия, частота и напряжение имеют разные
            единицы измерения.
            """
        ),
        code(
            """
            X = features_df[signal_features]
            y = features_df["condition_class"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
            )

            pd_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=max_iter, class_weight="balanced")),
            ])
            pd_model.fit(X_train, y_train)
            y_pred = pd_model.predict(X_test)

            print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
            print("Macro-F1:", round(f1_score(y_test, y_pred, average="macro"), 4))
            print(classification_report(y_test, y_pred))
            """
        ),
        code(
            """
            cm = confusion_matrix(y_test, y_pred, labels=pd_model.classes_)
            ConfusionMatrixDisplay(cm, display_labels=pd_model.classes_).plot(xticks_rotation=35)
            plt.title("Матрица ошибок классификации сигналов")
            plt.show()
            """
        ),
        md(
            """
            ## Сравнение с ансамблевой моделью

            Случайный лес (Random Forest) строит множество деревьев решений и
            усредняет их ответы. Для начального уровня важно не стремиться к
            максимальной сложности, а сравнить качество и интерпретируемость.
            """
        ),
        code(
            """
            forest_model = RandomForestClassifier(
                n_estimators=160,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )
            forest_model.fit(X_train, y_train)
            forest_pred = forest_model.predict(X_test)
            model_summary = pd.DataFrame([
                {
                    "model": "LogisticRegression",
                    "accuracy": accuracy_score(y_test, y_pred),
                    "macro_f1": f1_score(y_test, y_pred, average="macro"),
                },
                {
                    "model": "RandomForestClassifier",
                    "accuracy": accuracy_score(y_test, forest_pred),
                    "macro_f1": f1_score(y_test, forest_pred, average="macro"),
                },
            ])
            display(model_summary.round(4))
            """
        ),
        md(
            """
            ## АНТИПРИМЕР: утечка диагностической разметки

            Блок ниже показывает запрещенный прием. Если присоединить
            `state_code` из diagnostics-CSV, модель получает почти прямое
            числовое представление ответа. Такой результат нельзя использовать
            в отчете как качество диагностической модели.
            """
        ),
        code(
            """
            leakage_df = features_df.merge(diagnostics_df, on="sample_id", validate="one_to_one")
            leakage_features = signal_features + ["state_code"]
            X_leak = leakage_df[leakage_features]
            y_leak = leakage_df["condition_class"]
            X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
                X_leak, y_leak, test_size=0.25, random_state=RANDOM_STATE, stratify=y_leak
            )
            leakage_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
            leakage_model.fit(X_train_l, y_train_l)
            leakage_pred = leakage_model.predict(X_test_l)
            print("Запрещенный результат с утечкой, Macro-F1:", round(f1_score(y_test_l, leakage_pred, average="macro"), 4))
            """
        ),
        code(
            todo_text(
                teacher,
                "Энергия и число импульсных превышений лучше отделяют частые разряды от нормального режима, но внешняя помеха требует анализа спектра.",
                "signal_interpretation",
                "Сформулируйте один диагностический вывод по временным и частотным графикам.",
            )
        ),
        *source_section_07_09(),
        md(
            """
            ## Контрольный чек-лист отчета

            1. Указаны частота дискретизации и длительность окна.
            2. Построены временные графики и спектры.
            3. Объяснены амплитуда, энергия, SNR и доминирующая частота.
            4. Диагностические столбцы не использованы как признаки.
            5. Приведены метрики классификации и инженерный вывод.
            """
        ),
    ]
    return cells


def build_practice_08(teacher: bool) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = [
        md(
            f"""
            # Практическое занятие 8. Расчет режима энергосистемы в pandapower

            Версия: {"преподавательская" if teacher else "студенческая"}.

            Цель занятия - выполнить расчет установившегося режима
            электроэнергетической системы и интерпретировать напряжения шин,
            загрузку линий и изменение режима при росте нагрузки.
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_08_power_flow_features.csv",
                "practice_08_power_flow_diagnostics.csv",
                "practice_08_power_flow_scenarios.csv",
                "practice_07_09_dataset_catalog.csv",
                "practice_07_09_dataset_assignments.csv",
            ]
        ),
        code(COMMON_IMPORTS),
        md(
            """
            ## Теоретический блок

            Расчет режима энергосистемы (power flow calculation, load flow
            calculation) определяет модули и углы напряжений в шинах, потоки
            активной и реактивной мощности, а также загрузку линий. AC power
            flow - расчет переменного тока, в котором учитываются активная
            мощность `P`, реактивная мощность `Q`, модуль напряжения `U` и
            фазовый угол.

            Баланс мощности в узле:

            $$
            P_{gen} - P_{load} = P_{injected},\\quad
            Q_{gen} - Q_{load} = Q_{injected}.
            $$
            """
        ),
        md(
            """
            ## Последовательность работы

            1. Загрузить сценарии нагрузки и генерации.
            2. Создать учебную 6-узловую сеть 110 kV в `pandapower`.
            3. Выполнить AC power flow для базового сценария.
            4. Сравнить базовый режим и режим с увеличенной нагрузкой.
            5. Найти слабый узел и наиболее нагруженную линию.
            6. Объяснить, почему расчет режима не является заменой машинному
               обучению.
            """
        ),
        code(
            """
            features_df = pd.read_csv(DATA_DIR / "practice_08_power_flow_features.csv")
            diagnostics_df = pd.read_csv(DATA_DIR / "practice_08_power_flow_diagnostics.csv")
            scenarios_df = pd.read_csv(DATA_DIR / "practice_08_power_flow_scenarios.csv")

            print("features:", features_df.shape)
            print("diagnostics:", diagnostics_df.shape)
            print("scenarios:", scenarios_df.shape)
            display(scenarios_df.head())
            display(features_df.head())
            """
        ),
        md(
            """
            ## Паспорт сети

            Учебная сеть содержит 6 шин напряжением 110 kV, внешнюю сеть,
            один генератор и четыре нагрузки. Одна строка сценариев задает
            входы расчета. Столбцы с напряжениями и загрузкой линий являются
            результатами расчета, а не исходными признаками.
            """
        ),
        code(
            """
            display(features_df.groupby("scenario_label").agg(
                n=("scenario_id", "count"),
                mean_load=("total_load_mw", "mean"),
                mean_min_vm=("min_vm_pu", "mean"),
                mean_max_loading=("max_line_loading_percent", "mean"),
            ).round(4))
            """
        ),
        md(
            """
            ## Расчет базового режима в pandapower

            В этой ячейке используется та же функция построения сети, что и при
            генерации CSV. Это позволяет студенту увидеть объект `pandapower`
            и результат `runpp`, не переписывая большой код создания сети.
            """
        ),
        code(
            """
            from appai_lab.data_generators import _create_power_flow_network
            import pandapower as pp

            base_scenario = scenarios_df.iloc[0].to_dict()
            net = _create_power_flow_network(base_scenario)
            pp.runpp(net, algorithm="nr", init="flat", numba=False)
            display(net.bus[["name", "vn_kv"]])
            display(net.res_bus[["vm_pu", "va_degree"]].round(4))
            display(net.res_line[["p_from_mw", "q_from_mvar", "loading_percent"]].round(3))
            """
        ),
        md(
            """
            ## Визуальный анализ сценариев

            Диаграммы ниже показывают, как рост суммарной нагрузки связан со
            снижением минимального напряжения и ростом загрузки наиболее
            нагруженной линии.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].scatter(features_df["total_load_mw"], features_df["min_vm_pu"], alpha=0.75)
            axes[0].set_xlabel("Суммарная нагрузка, MW")
            axes[0].set_ylabel("Минимальное напряжение, p.u.")
            axes[0].set_title("Нагрузка и слабый узел")

            axes[1].scatter(features_df["total_load_mw"], features_df["max_line_loading_percent"], alpha=0.75, color="#c44e52")
            axes[1].axhline(100.0, color="black", linestyle="--", linewidth=1)
            axes[1].set_xlabel("Суммарная нагрузка, MW")
            axes[1].set_ylabel("Максимальная загрузка линии, percent")
            axes[1].set_title("Нагрузка и критическая линия")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            todo_value(
                teacher,
                1.20,
                "load_multiplier",
                "Задайте множитель нагрузки в диапазоне 1.10..1.30 для сценария чувствительности.",
            )
        ),
        code(
            todo_value(
                teacher,
                0.97,
                "voltage_limit_pu",
                "Задайте нижний учебный предел напряжения в диапазоне 0.95..0.98 p.u.",
            )
        ),
        md(
            """
            ## Сравнение базового и измененного режима

            Изменение нагрузки применяется к одному расчетному сценарию. Это
            не обучение модели, а инженерный численный эксперимент.
            """
        ),
        code(
            """
            changed = dict(base_scenario)
            for bus in [2, 3, 4, 5]:
                changed[f"load_bus_{bus}_mw"] *= load_multiplier
                changed[f"load_bus_{bus}_mvar"] *= load_multiplier

            changed_net = _create_power_flow_network(changed)
            pp.runpp(changed_net, algorithm="nr", init="flat", numba=False)

            comparison = pd.DataFrame({
                "bus": net.bus["name"].to_numpy(),
                "base_vm_pu": net.res_bus["vm_pu"].to_numpy(),
                "changed_vm_pu": changed_net.res_bus["vm_pu"].to_numpy(),
            })
            comparison["delta_vm_pu"] = comparison["changed_vm_pu"] - comparison["base_vm_pu"]
            comparison["below_limit_after_change"] = comparison["changed_vm_pu"] < voltage_limit_pu
            display(comparison.round(5))

            line_comparison = pd.DataFrame({
                "line": net.line["name"].to_numpy(),
                "base_loading_percent": net.res_line["loading_percent"].to_numpy(),
                "changed_loading_percent": changed_net.res_line["loading_percent"].to_numpy(),
            })
            line_comparison["delta_loading_percent"] = (
                line_comparison["changed_loading_percent"] - line_comparison["base_loading_percent"]
            )
            display(line_comparison.round(3))
            """
        ),
        code(
            """
            plt.figure(figsize=(9, 4))
            x = np.arange(len(comparison))
            plt.plot(x, comparison["base_vm_pu"], marker="o", label="Базовый режим")
            plt.plot(x, comparison["changed_vm_pu"], marker="o", label="Измененный режим")
            plt.xticks(x, comparison["bus"], rotation=20)
            plt.ylabel("Напряжение, p.u.")
            plt.title("Напряжения шин до и после роста нагрузки")
            plt.legend()
            plt.show()

            line_comparison.plot(x="line", y=["base_loading_percent", "changed_loading_percent"], kind="bar", figsize=(10, 4))
            plt.ylabel("Загрузка линии, percent")
            plt.title("Загрузка линий до и после роста нагрузки")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## АНТИПРИМЕР: расчет режима не является задачей машинного обучения

            Нельзя подменять физический расчет простым подбором зависимости без
            понимания входов и ограничений сети. Машинное обучение может быть
            полезным дополнением, но базовый режим должен проверяться расчетной
            моделью и инженерными ограничениями.
            """
        ),
        code(
            todo_text(
                teacher,
                "При росте нагрузки минимальное напряжение снижается, а наиболее загруженной остается линия 1; это указывает на чувствительность головного участка сети.",
                "power_flow_interpretation",
                "Опишите слабый узел и наиболее нагруженную линию.",
            )
        ),
        *source_section_07_09(),
        md(
            """
            ## Контрольный чек-лист отчета

            1. Описаны элементы сети и входные параметры сценария.
            2. Приведены таблицы напряжений шин и загрузки линий.
            3. Построены графики сравнения базового и измененного режима.
            4. Указаны слабый узел и критическая линия.
            5. Объяснено отличие расчета режима от модели машинного обучения.
            """
        ),
    ]
    return cells


def build_practice_09(teacher: bool) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = [
        md(
            f"""
            # Практическое занятие 9. Сравнительный анализ методов расчета режима энергосистемы

            Версия: {"преподавательская" if teacher else "студенческая"}.

            Цель занятия - сравнить AC power flow, DC power flow и простую
            суррогатную модель машинного обучения. Суррогатная модель
            (surrogate model) является приближением расчетной модели и не
            заменяет проверку физических ограничений.
            """
        ),
        *colab_bootstrap_cells(
            [
                "practice_09_power_flow_comparison_features.csv",
                "practice_09_power_flow_comparison_diagnostics.csv",
                "practice_09_power_flow_comparison.csv",
                "practice_07_09_dataset_catalog.csv",
                "practice_07_09_dataset_assignments.csv",
            ]
        ),
        code(COMMON_IMPORTS),
        md(
            """
            ## Теоретический блок

            AC power flow учитывает активные и реактивные мощности, модули и
            углы напряжений. DC power flow - линейное приближение, в котором
            обычно предполагаются малые углы, близкие к номинальным напряжения
            и малые потери. Поэтому DC-расчет полезен для быстрого анализа
            активных перетоков, но не является универсальной заменой AC-расчету.

            Ошибка сравнения:

            $$
            e_l = P^{AC}_l - P^{DC}_l.
            $$

            Здесь `l` - номер линии сети.
            """
        ),
        code(
            """
            features_df = pd.read_csv(DATA_DIR / "practice_09_power_flow_comparison_features.csv")
            diagnostics_df = pd.read_csv(DATA_DIR / "practice_09_power_flow_comparison_diagnostics.csv")
            full_df = pd.read_csv(DATA_DIR / "practice_09_power_flow_comparison.csv")

            print("features:", features_df.shape)
            print("diagnostics:", diagnostics_df.shape)
            print("full:", full_df.shape)
            display(features_df.head())
            display(diagnostics_df[[
                "scenario_id", "mean_abs_line_error_mw",
                "max_abs_line_error_mw", "mean_relative_line_error_percent"
            ]].head())
            """
        ),
        md(
            """
            ## Сравнение AC- и DC-перетоков

            На графике каждая точка соответствует одной линии одного сценария.
            Чем ближе точка к диагонали, тем меньше расхождение между
            нелинейным AC-расчетом и линейным DC-приближением.
            """
        ),
        code(
            """
            ac_cols = [f"ac_line_{i}_p_mw" for i in range(1, 8)]
            dc_cols = [f"dc_line_{i}_p_mw" for i in range(1, 8)]
            ac_values = diagnostics_df[ac_cols].to_numpy().ravel()
            dc_values = diagnostics_df[dc_cols].to_numpy().ravel()

            plt.figure(figsize=(6, 6))
            plt.scatter(ac_values, dc_values, alpha=0.55)
            lo = min(ac_values.min(), dc_values.min())
            hi = max(ac_values.max(), dc_values.max())
            plt.plot([lo, hi], [lo, hi], color="black", linestyle="--")
            plt.xlabel("AC power flow, MW")
            plt.ylabel("DC power flow, MW")
            plt.title("Сравнение перетоков активной мощности")
            plt.show()
            """
        ),
        code(
            """
            error_summary = diagnostics_df[[
                "mean_abs_line_error_mw",
                "max_abs_line_error_mw",
                "mean_relative_line_error_percent",
            ]].describe().round(4)
            display(error_summary)

            diagnostics_df["max_abs_line_error_mw"].hist(bins=18)
            plt.xlabel("Максимальная ошибка по линиям, MW")
            plt.ylabel("Число сценариев")
            plt.title("Распределение максимальной ошибки DC-приближения")
            plt.show()
            """
        ),
        md(
            """
            ## Безопасные признаки суррогатной модели

            Суррогатная модель должна получать только входы сценария: нагрузки,
            реактивные нагрузки, уставку генератора и масштаб нагрузки. В нее
            нельзя включать AC-перетоки и уже рассчитанные напряжения из той же
            строки, иначе получится утечка данных.
            """
        ),
        code(
            todo_list(
                teacher,
                [
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
                ],
                "surrogate_features",
                "Выберите входные признаки сценария без AC-результатов.",
            )
            + "\nforbidden_surrogate_features = {'min_vm_pu', 'max_line_loading_percent', 'total_line_loss_mw', 'critical_line', 'weak_bus'}\n"
            + "forbidden_surrogate_features |= {f'ac_line_{i}_p_mw' for i in range(1, 8)}\n"
            + "forbidden_surrogate_features |= {f'dc_line_{i}_p_mw' for i in range(1, 8)}\n"
            + "leaked = forbidden_surrogate_features & set(surrogate_features)\n"
            + "if leaked:\n"
            + "    raise ValueError(f'Обнаружена утечка данных: {sorted(leaked)}')\n"
        ),
        code(
            todo_value(
                teacher,
                "max_line_loading_percent",
                "surrogate_target",
                "Выберите целевой AC-показатель: 'max_line_loading_percent' или 'min_vm_pu'.",
            )
            + "\nif surrogate_target not in {'max_line_loading_percent', 'min_vm_pu'}:\n"
            + "    raise ValueError(\"surrogate_target должен быть 'max_line_loading_percent' или 'min_vm_pu'.\")\n"
        ),
        code(
            todo_value(
                teacher,
                6,
                "surrogate_max_depth",
                "Задайте max_depth случайного леса в диапазоне 3..10.",
            )
        ),
        md(
            """
            ## Обучение суррогатной модели

            В этом примере прогнозируется расчетный AC-показатель по входным
            параметрам сценария. Метрики MAE, RMSE и R2 показывают точность
            приближения, но не отменяют необходимости контрольного AC-расчета.
            """
        ),
        code(
            """
            X = features_df[surrogate_features]
            y = features_df[surrogate_target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=RANDOM_STATE
            )

            surrogate_model = RandomForestRegressor(
                n_estimators=180,
                max_depth=surrogate_max_depth,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
            )
            surrogate_model.fit(X_train, y_train)
            y_pred = surrogate_model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": rmse,
                "R2": r2_score(y_test, y_pred),
            }
            display(pd.DataFrame([metrics]).round(4))
            """
        ),
        code(
            """
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred, alpha=0.75)
            lo = min(y_test.min(), y_pred.min())
            hi = max(y_test.max(), y_pred.max())
            plt.plot([lo, hi], [lo, hi], color="black", linestyle="--")
            plt.xlabel("AC-расчет")
            plt.ylabel("Прогноз суррогатной модели")
            plt.title(f"Прогноз целевого показателя: {surrogate_target}")
            plt.show()

            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.75)
            plt.axhline(0.0, color="black", linestyle="--")
            plt.xlabel("Прогноз")
            plt.ylabel("Остаток")
            plt.title("Остатки суррогатной модели")
            plt.show()
            """
        ),
        md(
            """
            ## АНТИПРИМЕР: использование AC-результатов как входов

            Если добавить `ac_line_1_p_mw` и другие результаты AC-расчета в
            признаки, модель получает информацию, вычисленную той же физической
            моделью. Это искусственно завышает качество и не является честной
            суррогатной моделью.
            """
        ),
        code(
            """
            leakage_df = features_df.merge(diagnostics_df, on="scenario_id", validate="one_to_one")
            leakage_features = surrogate_features + ["ac_line_1_p_mw", "ac_line_2_p_mw", "total_line_loss_mw"]
            X_leak = leakage_df[leakage_features]
            y_leak = leakage_df[surrogate_target]
            X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
                X_leak, y_leak, test_size=0.25, random_state=RANDOM_STATE
            )
            leakage_model = RandomForestRegressor(n_estimators=180, random_state=RANDOM_STATE)
            leakage_model.fit(X_train_l, y_train_l)
            leakage_pred = leakage_model.predict(X_test_l)
            print("Запрещенный R2 с AC-утечкой:", round(r2_score(y_test_l, leakage_pred), 4))
            """
        ),
        code(
            todo_text(
                teacher,
                "DC-приближение дает малую среднюю ошибку для учебной transmission-like сети, но максимальная ошибка по отдельным линиям требует отдельного контроля.",
                "comparison_interpretation",
                "Сформулируйте вывод о границах применимости DC-приближения и суррогатной модели.",
            )
        ),
        *source_section_07_09(),
        md(
            """
            ## Контрольный чек-лист отчета

            1. Объяснены отличия AC power flow и DC power flow.
            2. Рассчитаны MAE/RMSE или абсолютные ошибки по линиям.
            3. Построено сравнение AC и DC перетоков.
            4. Суррогатная модель обучена только на входах сценария.
            5. Антипример утечки данных явно отделен от основного результата.
            """
        ),
    ]
    return cells


def build_all() -> None:
    notebooks = [
        ("07_pd_signal_analysis", build_practice_07),
        ("08_pandapower_power_flow", build_practice_08),
        ("09_power_flow_comparison", build_practice_09),
    ]
    for stem, builder in notebooks:
        save_notebook(builder(False), NOTEBOOKS_STUDENT / f"{stem}_student.ipynb")
        save_notebook(builder(True), NOTEBOOKS_TEACHER / f"{stem}_teacher.ipynb")


if __name__ == "__main__":
    build_all()

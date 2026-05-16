# Лекционные презентации курса

Презентации построены на Marp (Markdown → PPTX / PDF / HTML) и
повторяют структуру практических занятий 1–9 из [PLAN.md](../PLAN.md).
Каждая лекция содержит теоретическую часть из PLAN.md, готовые графики
из соответствующего преподавательского ноутбука, фрагменты кода,
антипримеры (утечка данных, переобучение) и чек-лист критериев зачёта.

## Состав

| Файл | Лекция | Длительность |
|---|---|---|
| [01_engineering_data.md](01_engineering_data.md) | Инженерные данные и постановка задачи | 90 мин |
| [02_motor_regression.md](02_motor_regression.md) | Регрессия КПД и утечка данных | 90 мин |
| [03_drive_decision_tree.md](03_drive_decision_tree.md) | Дерево решений и инженерная безопасность | 90 мин |
| [04_haps_thermal_modeling.md](04_haps_thermal_modeling.md) | Прогноз температуры электропривода HAPS | 90 мин |
| [05_partial_discharge_classification.md](05_partial_discharge_classification.md) | Классификация частичных разрядов | 90 мин |
| [06_equipment_modes_clustering.md](06_equipment_modes_clustering.md) | Кластеризация режимов оборудования | 90 мин |
| [07_pd_signal_analysis.md](07_pd_signal_analysis.md) | Анализ сигналов частичных разрядов | 90 мин |
| [08_pandapower_power_flow.md](08_pandapower_power_flow.md) | Расчет режима энергосистемы в pandapower | 90 мин |
| [09_power_flow_comparison.md](09_power_flow_comparison.md) | Сравнение методов расчета режима | 90 мин |
| [theme.css](theme.css) | Marp-тема курса | — |
| [img/](img/) | Графики, извлечённые из teacher-ноутбуков | — |

## Установка зависимостей

```bash
# marp-cli (для всех форматов)
npm install --global @marp-team/marp-cli

# nbformat для извлечения PNG из .ipynb (уже в requirements-base.txt)
pip install nbformat
```

**Для PDF и PPTX** marp-cli использует Chrome/Chromium/Firefox для рендеринга.
В Linux/WSL установить:

```bash
sudo apt-get install -y chromium-browser
# либо
sudo apt-get install -y firefox
```

Альтернатива без sudo — установить локальный браузер через `puppeteer`:

```bash
npx -y puppeteer browsers install chrome
export CHROME_PATH="$HOME/.cache/puppeteer/chrome/$(ls ~/.cache/puppeteer/chrome)/chrome-linux64/chrome"
```

HTML собирается без браузера, поэтому базовая сборка работает сразу.

## Сборка

Только HTML (работает без браузера, рекомендуется для первого прогона):

```bash
python scripts/build_lecture_slides.py --formats html
```

Полный цикл (PDF + PPTX + HTML, все реализованные лекции):

```bash
python scripts/build_lecture_slides.py
```

Только PDF и одна лекция:

```bash
python scripts/build_lecture_slides.py --formats pdf --lecture 01_engineering_data
```

Скрипт сначала обновляет PNG-графики (`scripts/export_teacher_figures.py`),
затем вызывает `marp-cli` с темой `theme.css`.

## Предварительный просмотр

Marp поддерживает live-preview:

```bash
cd docs/slides
marp 01_engineering_data.md --watch --html --output 01_engineering_data.html
```

Открыть `01_engineering_data.html` в браузере — изменения markdown
обновляют слайды автоматически.

## Заметки лектора (speaker notes)

В Marp speaker notes — это HTML-комментарии `<!-- ... -->` внутри слайда.
При экспорте в PDF ключ `--pdf-notes` (включён в `build_lecture_slides.py`)
добавляет заметки на отдельные страницы после каждого слайда.
В PowerPoint заметки попадают в стандартное поле speaker notes.

В режиме presenter (`p` в Marp-viewer) заметки видны только лектору.

## Обновление графиков

Преподавательские ноутбуки `notebooks/teacher/*.ipynb` хранятся с
выполненными ячейками. При изменении генератора данных или кода
ноутбука нужно перевыполнить ноутбук и затем:

```bash
python scripts/export_teacher_figures.py
```

PNG-файлы перезаписываются в `docs/slides/img/` по детерминированным
именам.

## Идеология слайдов

- **Гибридный формат**: на слайдах — тезисы, формулы, графики и
  ключевые таблицы. Расширенные комментарии вынесены в speaker notes.
- **Антипримеры выделены** красной темой (`<!-- _class: warning -->`).
  Демонстрация утечки данных снабжена явной пометкой «не использовать
  в отчёте».
- **Чек-листы зачёта** выделены зелёной темой (`<!-- _class: checklist -->`).
- **Связь с шаблоном отчёта**: каждый блок завершается ссылкой на
  пункт, который попадает в `docs/templates/practice_report_template.md`.

## Расширение на занятия 10–12

Сейчас реализованы 9 из 12 запланированных лекций. Шаблон одной
презентации (12–22 слайда):

1. Титул
2. Цель занятия
3. Структура занятия (тайминг)
4. Раздел: теоретическое введение
5. Определения / формулы
6. Метрики или метод
7. Связь признаков с физикой
8. Раздел: демонстрация шаблона
9. Структура данных
10. Раздел: практическое задание
11. Пошаговый план
12. Графики из teacher-ноутбука
13. Антипример
14. Типовые ошибки
15. Чек-лист критериев зачёта
16. Контрольные вопросы
17. Итог занятия

Для занятий 10–12 потребуется сначала собрать соответствующие
preview-ноутбуки и графики, затем создать `10_*.md ... 12_*.md` по тому
же шаблону.

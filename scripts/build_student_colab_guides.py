"""Сборка PDF-инструкций для студентов по работе в Google Colab.

Исходным редактируемым форматом являются Markdown-файлы в каталоге ``docs``.
Основной способ сборки PDF - ``pandoc`` с движком XeLaTeX. Такой способ
корректно обрабатывает Markdown-таблицы, ссылки, переносы строк и кириллицу,
поэтому предпочтителен для учебно-методических материалов.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class PandocDocument:
    """Описание документа, собираемого из Markdown в PDF."""

    source: Path
    target: Path
    title: str
    subtitle: str
    toc: bool
    title_block: bool = True
    document_class: str = "article"
    font_size: str = "10pt"
    margin: str = "18mm"


DOCUMENTS = [
    PandocDocument(
        source=DOCS_DIR / "student_colab_guide.md",
        target=DOCS_DIR / "student_colab_guide.pdf",
        title="Инструкция для студентов по запуску практических блокнотов в Google Colab",
        subtitle="Расширенная версия",
        toc=True,
    ),
    PandocDocument(
        source=DOCS_DIR / "student_colab_quickstart.md",
        target=DOCS_DIR / "student_colab_quickstart.pdf",
        title="Краткая памятка по запуску блокнотов в Google Colab",
        subtitle="Одностраничная версия для занятия",
        toc=False,
        title_block=False,
        document_class="extarticle",
        font_size="9pt",
        margin="10mm",
    ),
]


def _require_binary(name: str) -> None:
    """Проверить наличие внешней программы."""

    if shutil.which(name) is None:
        raise RuntimeError(
            f"Не найдена программа `{name}`. Для сборки PDF установите pandoc "
            "и TeX-движок XeLaTeX."
        )


def build_pdf_with_pandoc(document: PandocDocument) -> None:
    """Собрать один PDF-документ через pandoc и XeLaTeX."""

    document.target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pandoc",
        str(document.source),
        "--from",
        "markdown+pipe_tables+fenced_code_blocks",
        "--to",
        "pdf",
        "--pdf-engine=xelatex",
        "--standalone",
        "--metadata",
        "lang=ru-RU",
        "--variable",
        "mainfont=DejaVu Serif",
        "--variable",
        "sansfont=DejaVu Sans",
        "--variable",
        "monofont=DejaVu Sans Mono",
        "--variable",
        f"documentclass={document.document_class}",
        "--variable",
        f"fontsize={document.font_size}",
        "--variable",
        "geometry:a4paper",
        "--variable",
        f"geometry:margin={document.margin}",
        "--variable",
        "colorlinks=true",
        "--variable",
        "linkcolor=blue",
        "--variable",
        "urlcolor=blue",
        "--variable",
        "toccolor=blue",
        "-o",
        str(document.target),
    ]
    if document.title_block:
        command[8:8] = [
            "--metadata",
            f"title={document.title}",
            "--metadata",
            f"subtitle={document.subtitle}",
        ]
    if document.toc:
        command.insert(8, "--toc")
        command.insert(9, "--toc-depth=2")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    _require_binary("pandoc")
    _require_binary("xelatex")
    for document in DOCUMENTS:
        build_pdf_with_pandoc(document)
        print(document.target.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

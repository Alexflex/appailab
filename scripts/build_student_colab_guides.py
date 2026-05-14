"""Сборка PDF-инструкций для студентов по работе в Google Colab.

Скрипт читает Markdown-файлы из каталога docs и создает PDF-версии. Для
генерации используется matplotlib, так как эта зависимость уже входит в
базовое окружение курса. PDF-файлы являются производными артефактами, а
исходным редактируемым форматом остаются Markdown-документы.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class PdfStyle:
    """Параметры верстки PDF-страницы."""

    title_size: int = 15
    h1_size: int = 14
    h2_size: int = 11
    body_size: int = 8
    code_size: int = 7
    left: float = 0.08
    right: float = 0.94
    top: float = 0.94
    bottom: float = 0.07
    line_step: float = 0.020
    paragraph_gap: float = 0.010


def _normalize_links(text: str) -> str:
    """Заменить Markdown-ссылки на читаемый текст для PDF."""

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", text)


def _prepare_markdown_lines(markdown_text: str) -> list[tuple[str, str]]:
    """Преобразовать Markdown в последовательность типизированных строк.

    Возвращаемые типы:
    - title: главный заголовок документа;
    - h2: заголовок второго уровня;
    - body: обычный текст;
    - code: кодовые блоки и таблицы.
    """

    result: list[tuple[str, str]] = []
    in_code = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line:
            result.append(("blank", ""))
            continue
        if in_code:
            result.append(("code", line))
            continue
        if line.startswith("# "):
            result.append(("title", line[2:].strip()))
        elif line.startswith("## "):
            result.append(("h2", line[3:].strip()))
        elif line.startswith("### "):
            result.append(("h2", line[4:].strip()))
        elif line.startswith("|"):
            result.append(("code", _normalize_links(line)))
        else:
            result.append(("body", _normalize_links(line)))
    return result


def _wrap_line(kind: str, text: str) -> list[str]:
    """Разбить строку на переносимые фрагменты."""

    if kind == "title":
        width = 48
    elif kind == "h2":
        width = 82
    elif kind == "code":
        width = 84
    else:
        width = 98
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=kind == "code",
        break_on_hyphens=kind == "code",
    ) or [""]


def build_pdf(markdown_path: Path, pdf_path: Path, *, style: PdfStyle | None = None) -> None:
    """Собрать PDF из Markdown-файла."""

    style = style or PdfStyle()
    lines = _prepare_markdown_lines(markdown_path.read_text(encoding="utf-8"))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        page_number = 0
        fig = None
        ax = None
        y = style.top

        def new_page() -> None:
            nonlocal fig, ax, y, page_number
            if fig is not None:
                ax.text(
                    0.5,
                    0.035,
                    f"Страница {page_number}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    family="DejaVu Sans",
                    color="#555555",
                )
                pdf.savefig(fig)
                plt.close(fig)
            page_number += 1
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.set_axis_off()
            y = style.top

        def ensure_space(required_lines: int) -> None:
            nonlocal y
            required_height = required_lines * style.line_step + style.paragraph_gap
            if y - required_height < style.bottom:
                new_page()

        new_page()
        for kind, text in lines:
            if kind == "blank":
                y -= style.paragraph_gap
                continue

            wrapped = _wrap_line(kind, text)
            ensure_space(len(wrapped) + (1 if kind in {"title", "h2"} else 0))

            if kind == "title":
                fontsize = style.title_size
                weight = "bold"
                color = "#1f2933"
                family = "DejaVu Sans"
            elif kind == "h2":
                fontsize = style.h2_size
                weight = "bold"
                color = "#1f2933"
                family = "DejaVu Sans"
            elif kind == "code":
                fontsize = style.code_size
                weight = "normal"
                color = "#333333"
                family = "DejaVu Sans Mono"
            else:
                fontsize = style.body_size
                weight = "normal"
                color = "#111111"
                family = "DejaVu Sans"

            for index, part in enumerate(wrapped):
                prefix = ""
                if kind == "body" and index == 0:
                    prefix = ""
                ax.text(
                    style.left,
                    y,
                    prefix + part,
                    ha="left",
                    va="top",
                    fontsize=fontsize,
                    fontweight=weight,
                    family=family,
                    color=color,
                    transform=ax.transAxes,
                )
                y -= style.line_step

            if kind in {"title", "h2"}:
                y -= style.paragraph_gap

        if fig is not None:
            ax.text(
                0.5,
                0.035,
                f"Страница {page_number}",
                ha="center",
                va="center",
                fontsize=7,
                family="DejaVu Sans",
                color="#555555",
            )
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    build_pdf(
        DOCS_DIR / "student_colab_guide.md",
        DOCS_DIR / "student_colab_guide.pdf",
    )
    build_pdf(
        DOCS_DIR / "student_colab_quickstart.md",
        DOCS_DIR / "student_colab_quickstart.pdf",
        style=PdfStyle(
            title_size=13,
            h1_size=12,
            h2_size=9,
            body_size=7,
            code_size=6,
            line_step=0.015,
            paragraph_gap=0.004,
        ),
    )
    print("PDF-инструкции для студентов созданы.")


if __name__ == "__main__":
    main()

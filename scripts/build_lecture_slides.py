"""Сборка лекционных презентаций курса.

Шаги:
1. Извлекает PNG из преподавательских ноутбуков в docs/slides/img/.
2. Конвертирует Marp-Markdown в PDF, PPTX и HTML через marp-cli.

Требования:
    npm install --global @marp-team/marp-cli
    pip install nbformat

Запуск:
    python scripts/build_lecture_slides.py
    python scripts/build_lecture_slides.py --formats pdf html
    python scripts/build_lecture_slides.py --lecture 01
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = PROJECT_ROOT / "docs/slides"
THEME_FILE = SLIDES_DIR / "theme.css"
LECTURES = (
    "01_engineering_data",
    "02_motor_regression",
    "03_drive_decision_tree",
    "04_haps_thermal_modeling",
    "05_partial_discharge_classification",
    "06_equipment_modes_clustering",
)
DEFAULT_FORMATS = ("pdf", "pptx", "html")


def _ensure_marp() -> str:
    marp_path = shutil.which("marp")
    if not marp_path:
        sys.exit(
            "marp-cli не найден. Установите: npm install --global @marp-team/marp-cli"
        )
    return marp_path


def _refresh_figures() -> None:
    script = PROJECT_ROOT / "scripts/export_teacher_figures.py"
    print("[1/2] Извлечение графиков из teacher-блокнотов")
    subprocess.run([sys.executable, str(script)], check=True)


def _convert(marp_bin: str, source: Path, fmt: str) -> None:
    target = source.with_suffix(f".{fmt}")
    args = [
        marp_bin,
        str(source.name),
        f"--{fmt}",
        "--theme-set",
        str(THEME_FILE.name),
        "--allow-local-files",
        "--output",
        str(target.name),
    ]
    if fmt == "pdf":
        args.append("--pdf-notes")
    print(f"      → {target.name}")
    subprocess.run(args, check=True, cwd=SLIDES_DIR)


def build(formats: tuple[str, ...], lectures: tuple[str, ...]) -> None:
    marp_bin = _ensure_marp()
    _refresh_figures()
    print("[2/2] Сборка презентаций")
    for stem in lectures:
        source = SLIDES_DIR / f"{stem}.md"
        if not source.exists():
            print(f"      пропуск {stem}.md (нет файла)")
            continue
        print(f"   • {source.name}")
        for fmt in formats:
            _convert(marp_bin, source, fmt)
    print("Готово.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "pptx", "html"),
        default=list(DEFAULT_FORMATS),
        help="Выходные форматы (по умолчанию: pdf pptx html)",
    )
    parser.add_argument(
        "--lecture",
        action="append",
        choices=LECTURES,
        help="Собирать только указанные лекции (можно повторять)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ns = _parse_args()
    selected = tuple(ns.lecture) if ns.lecture else LECTURES
    build(tuple(ns.formats), selected)

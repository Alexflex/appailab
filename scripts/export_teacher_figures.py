"""Извлечение PNG-графиков из преподавательских Jupyter-блокнотов.

Назначение: достать встроенные выходные изображения из notebooks/teacher/*.ipynb
и сохранить их в docs/slides/img/ под детерминированными именами для
использования в Marp-презентациях.

Запуск:
    python scripts/export_teacher_figures.py

Файлы перезаписываются. Если в блокноте нет outputs (он не был выполнен),
скрипт выведет предупреждение.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "01": PROJECT_ROOT / "notebooks/teacher/01_engineering_data_teacher.ipynb",
    "02": PROJECT_ROOT / "notebooks/teacher/02_motor_regression_teacher.ipynb",
    "03": PROJECT_ROOT / "notebooks/teacher/03_drive_decision_tree_teacher.ipynb",
    "04": PROJECT_ROOT / "notebooks/teacher/04_haps_thermal_modeling_teacher.ipynb",
    "05": PROJECT_ROOT / "notebooks/teacher/05_partial_discharge_classification_teacher.ipynb",
    "06": PROJECT_ROOT / "notebooks/teacher/06_equipment_modes_clustering_teacher.ipynb",
}
OUTPUT_DIR = PROJECT_ROOT / "docs/slides/img"


def _slug(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text)
    return text[:max_len].strip("_") or "fig"


def _preceding_title(cells, index: int) -> str:
    for j in range(index - 1, -1, -1):
        if cells[j].cell_type == "markdown":
            md = "".join(cells[j].source).strip()
            heading = re.search(r"^#+\s*(.+)$", md, flags=re.MULTILINE)
            if heading:
                return heading.group(1).strip()
            return md.splitlines()[0][:80]
    return "figure"


def export() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[str]] = {}
    for tag, path in NOTEBOOKS.items():
        for old_image in OUTPUT_DIR.glob(f"{tag}_*.png"):
            old_image.unlink()
        nb = nbformat.read(path, as_version=4)
        counter = 0
        saved: list[str] = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                data = output.get("data") or {}
                png_b64 = data.get("image/png")
                if not png_b64:
                    continue
                counter += 1
                title = _preceding_title(nb.cells, i)
                fname = f"{tag}_{counter:02d}_{_slug(title)}.png"
                target = OUTPUT_DIR / fname
                target.write_bytes(base64.b64decode(png_b64))
                saved.append(fname)
        summary[tag] = saved
        if not saved:
            print(f"[warn] {path.name}: outputs не найдены, выполните блокнот.")
        else:
            print(f"[ok]  {path.name}: сохранено {len(saved)} рисунков")
    index = OUTPUT_DIR / "INDEX.txt"
    lines = []
    for tag, files in summary.items():
        lines.append(f"# Занятие {tag}")
        lines.extend(f"  {name}" for name in files)
        lines.append("")
    index.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    export()

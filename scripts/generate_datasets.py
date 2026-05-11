"""Создание CSV-файлов для практических занятий 1-3."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from appai_lab import create_all_datasets  # noqa: E402


def main() -> None:
    paths = create_all_datasets(PROJECT_ROOT / "data" / "processed")
    print("Созданы учебные наборы данных:")
    print(f"- {paths.practice_01}")
    print(f"- {paths.practice_02}")
    print(f"- {paths.practice_02_features}")
    print(f"- {paths.practice_02_diagnostics}")
    print(f"- {paths.practice_03}")
    print(f"- {paths.practice_03_features}")
    print(f"- {paths.practice_03_diagnostics}")
    print(f"- {paths.catalog}")
    print(f"- {paths.assignments}")
    print(f"- {paths.metadata}")


if __name__ == "__main__":
    main()

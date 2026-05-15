"""Создание CSV-файлов для практических занятий 1-6."""

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
    print(f"- {paths.practice_04}")
    print(f"- {paths.practice_04_features}")
    print(f"- {paths.practice_04_diagnostics}")
    print(f"- {paths.practice_05}")
    print(f"- {paths.practice_05_features}")
    print(f"- {paths.practice_05_diagnostics}")
    print(f"- {paths.practice_06}")
    print(f"- {paths.practice_06_features}")
    print(f"- {paths.practice_06_diagnostics}")
    print(f"- {paths.catalog}")
    print(f"- {paths.assignments}")
    print(f"- {paths.catalog_04_06}")
    print(f"- {paths.assignments_04_06}")
    print(f"- {paths.metadata}")


if __name__ == "__main__":
    main()

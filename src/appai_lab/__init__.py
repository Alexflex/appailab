"""Учебные материалы по дисциплине "Прикладной искусственный интеллект"."""

from .data_generators import (
    RANDOM_SEED,
    create_all_datasets,
    generate_drive_mode_classification,
    generate_motor_measurements,
)

__all__ = [
    "RANDOM_SEED",
    "create_all_datasets",
    "generate_drive_mode_classification",
    "generate_motor_measurements",
]

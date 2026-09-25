"""Reproducible Jev zero-shot versus few-shot evaluation on AG News."""

DATASET_NAME = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
LABELS = ("World", "Sports", "Business", "Sci/Tech")
TEST_PER_CLASS = 500
TEST_SEED = 20260925
DRAW_SEEDS = (0, 1, 2, 3, 4)
SHOT_LEVELS = (0, 4, 8, 16)

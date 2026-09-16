"""Shared pytest configuration.

Keeps the model suite offline by default: a test that quietly downloads timm
weights would hide exactly the failure mode this project has to survive (Kaggle
code competitions run with no internet).
"""

import os


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: builds a large backbone (>50M params); deselect with -m 'not slow'",
    )
    # Fail fast and loudly instead of hanging on a download attempt.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

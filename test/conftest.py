import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wathiqa_agent.config import Settings  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture()
def settings() -> Settings:
    return Settings()

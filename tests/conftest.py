import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth  # noqa: E402


@pytest.fixture
def project(tmp_path):
    """A two-night synthetic project with masters and flats."""
    return synth.make_project(tmp_path / "project")

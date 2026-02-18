from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


_BASE_TMP = Path("tests/py_tmp2")


@pytest.fixture
def tmp_path() -> Path:
    _BASE_TMP.mkdir(parents=True, exist_ok=True)
    path = _BASE_TMP / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)

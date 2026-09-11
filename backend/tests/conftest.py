import os
import tempfile
from pathlib import Path

import pytest

# Point the app at a throwaway database before it is imported.
_tmpdir = tempfile.mkdtemp(prefix="mi-sandbox-test-")
os.environ["MI_DB_PATH"] = str(Path(_tmpdir) / "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(main.app) as c:
        main.model_service.load()  # blocks until the background loader has finished
        yield c

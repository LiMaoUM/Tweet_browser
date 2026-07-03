import importlib
import os
import sys

import pytest

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "Frontend")
sys.path.insert(0, os.path.abspath(FRONTEND_DIR))


@pytest.fixture
def fresh_config(monkeypatch):
    """Reload the config module with a controlled environment, restoring defaults after."""
    import config

    def _load(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)

    yield _load
    monkeypatch.undo()
    importlib.reload(config)

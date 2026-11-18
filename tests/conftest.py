from __future__ import annotations

from pathlib import Path

import pytest

from rootline_mcp.config import Settings
from rootline_mcp.mock import build
from rootline_mcp.runner import MockBackend
from rootline_mcp.store import LocalSource, Store
from rootline_mcp.tools import Tools


@pytest.fixture()
def mock_root(tmp_path: Path) -> Path:
    return build(tmp_path / "data")


@pytest.fixture()
def store(mock_root: Path) -> Store:
    return Store(
        [LocalSource("published", mock_root / "published"), LocalSource("work", mock_root / "work")]
    )


@pytest.fixture()
def tools(store: Store) -> Tools:
    return Tools(Settings(mock=True), store, MockBackend(), runs_allowed=True)

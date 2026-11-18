from pathlib import Path

import pytest

from bigdb_mvp.data import civil_download


@pytest.fixture(autouse=True)
def use_isolated_download_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(civil_download, "DOWNLOAD_ROOT", tmp_path)
    monkeypatch.setattr(civil_download, "STAGING_ROOT", tmp_path)

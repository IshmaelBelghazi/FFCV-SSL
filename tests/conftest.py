"""Test-suite settings; library defaults are unchanged."""
import pytest

import ffcv.loader.loader
import ffcv.writer

# DatasetWriter and Loader default to one worker per CPU (num_workers=-1). Most tests write a tiny dataset,
# so forking a torch/numba-loaded interpreter cpu_count() times dominated each test's run time.
TEST_WORKERS = 2


@pytest.fixture(autouse=True)
def _few_default_workers(monkeypatch):
    monkeypatch.setattr(ffcv.writer, "cpu_count", lambda: TEST_WORKERS)
    monkeypatch.setattr(ffcv.loader.loader, "cpu_count", lambda: TEST_WORKERS)

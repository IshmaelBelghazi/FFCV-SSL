"""A crashing writer worker must fail the write instead of hanging the parent forever."""
from tempfile import NamedTemporaryFile

import pytest

from ffcv.fields import IntField
from ffcv.writer import DatasetWriter


class BrokenDataset:
    def __len__(self):
        return 20

    def __getitem__(self, index):
        if index == 13:
            raise RuntimeError("boom")
        return (index,)


class GoodDataset(BrokenDataset):
    def __getitem__(self, index):
        return (index,)


def test_worker_exception_raises_instead_of_hanging():
    with NamedTemporaryFile() as handle:
        writer = DatasetWriter(handle.name, {'label': IntField()}, num_workers=2)
        with pytest.raises(RuntimeError, match="worker"):
            writer.from_indexed_dataset(BrokenDataset(), chunksize=5)


def test_healthy_workers_still_write_every_sample():
    with NamedTemporaryFile() as handle:
        DatasetWriter(handle.name, {'label': IntField()}, num_workers=2).from_indexed_dataset(
            GoodDataset(), chunksize=5)

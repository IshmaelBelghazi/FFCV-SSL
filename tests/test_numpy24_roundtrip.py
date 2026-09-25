"""Small native/JIT round trip used when updating the supported NumPy minor."""

import numpy as np

from ffcv.fields import IntField, NDArrayField
from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter


class TinyDataset:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        index = int(index)
        return np.arange(4, dtype=np.float32) + np.float32(index), index


def test_numpy24_writer_loader_roundtrip(tmp_path):
    path = str(tmp_path / "tiny.beton")
    DatasetWriter(path, {"x": NDArrayField(shape=(4,), dtype=np.dtype("float32")), "y": IntField()}, num_workers=1).from_indexed_dataset(TinyDataset())
    loader = Loader(path, batch_size=4, num_workers=1, order=OrderOption.SEQUENTIAL, drop_last=False, pipelines={"x": [NDArrayDecoder()], "y": [IntDecoder()]})
    # Copy before the loader reuses its batch buffers.
    batches = [(x.copy(), y.copy()) for x, y in loader]
    x = np.concatenate([batch[0] for batch in batches])
    y = np.concatenate([batch[1] for batch in batches]).reshape(-1)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(x, np.stack([TinyDataset()[i][0] for i in range(8)]))
    np.testing.assert_array_equal(y, np.arange(8))

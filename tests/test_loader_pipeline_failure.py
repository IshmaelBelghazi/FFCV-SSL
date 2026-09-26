"""An exception inside the loading thread must reach the consumer instead of hanging it."""
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from ffcv.fields import IntField
from ffcv.fields.basics import IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.writer import DatasetWriter


class Explode(Operation):
    def generate_code(self):
        def explode(images, dst):
            raise RuntimeError("pipeline failure")
        return explode

    def declare_state_and_memory(self, previous_state):
        return previous_state, AllocationQuery((1,), np.dtype('<i8'))


class Numbers:
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return (index,)


@pytest.mark.parametrize('compiled', [False, True])
def test_pipeline_exception_reaches_the_consumer(compiled):
    with NamedTemporaryFile() as handle:
        DatasetWriter(handle.name, {'index': IntField()}, num_workers=1).from_indexed_dataset(Numbers())
        Compiler.set_enabled(compiled)
        loader = Loader(handle.name, batch_size=4, num_workers=1, order=OrderOption.SEQUENTIAL,
                        pipelines={'index': [IntDecoder(), Explode()]})
        with pytest.raises(RuntimeError, match="pipeline failure"):
            for _ in loader:
                pass

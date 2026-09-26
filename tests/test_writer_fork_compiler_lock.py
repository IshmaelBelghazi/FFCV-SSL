"""Forked writer workers must not inherit numba's compiler lock held by another thread."""
import threading
from tempfile import NamedTemporaryFile

from numba import njit
from numba.core.compiler_lock import global_compiler_lock

from ffcv.fields import IntField
from ffcv.writer import DatasetWriter


@njit
def _double(value):
    return 2 * value


class CompilesInWorker:
    """The first __getitem__ in a worker compiles `_double`, taking the compiler lock."""
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return (int(_double(index)) // 2,)


def test_workers_start_while_another_thread_is_compiling():
    # Another thread holds the lock when the writer forks (as a loader thread compiling
    # a pipeline does) and releases it half a second later.
    held, release = threading.Event(), threading.Event()

    def compile_in_background():
        with global_compiler_lock:
            held.set()
            release.wait(10)

    background = threading.Thread(target=compile_in_background)
    background.start()
    held.wait(10)
    threading.Timer(0.5, release.set).start()

    def write():
        with NamedTemporaryFile() as handle:
            DatasetWriter(handle.name, {'label': IntField()}, num_workers=2).from_indexed_dataset(
                CompilesInWorker(), chunksize=5)

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    writer.join(60)
    release.set()
    background.join()
    assert not writer.is_alive(), "writer workers deadlocked on an inherited compiler lock"

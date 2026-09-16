from __future__ import annotations

import ctypes
import re
import subprocess
from importlib.util import find_spec
from pathlib import Path


def native_extension() -> Path:
    package = find_spec("ffcv")
    if package is not None:
        for location in package.submodule_search_locations or ():
            extensions = list(Path(location).glob("_libffcv*.so"))
            if extensions:
                return extensions[0]
    raise FileNotFoundError("ffcv._libffcv is not installed")


def test_native_extension_links_only_required_opencv_modules() -> None:
    completed = subprocess.run(
        ["readelf", "-d", str(native_extension())],
        check=True,
        capture_output=True,
        text=True,
    )
    modules = set(re.findall(r"Shared library: \[libopencv_([^.]+)\.so", completed.stdout))

    assert modules == {"core", "imgproc"}


def test_native_extension_loads_with_current_system_libraries() -> None:
    ctypes.CDLL(str(native_extension()))

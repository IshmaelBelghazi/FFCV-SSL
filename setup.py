"""Native extension build; distribution metadata lives in pyproject.toml."""
import os
import shlex
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def environment_paths(name):
    return [path for path in os.environ.get(name, "").split(os.pathsep) if path]


def pkg_config_paths(package):
    try:
        completed = subprocess.run(
            ["pkg-config", "--cflags-only-I", "--libs-only-L", package],
            check=False, capture_output=True, text=True,
        )
    except FileNotFoundError:
        return [], []
    if completed.returncode != 0:
        return [], []
    tokens = shlex.split(completed.stdout)
    return ([token[2:] for token in tokens if token.startswith("-I")],
            [token[2:] for token in tokens if token.startswith("-L")])


def opencv_paths():
    include_dirs = environment_paths("FFCV_OPENCV_INCLUDE_DIRS")
    library_dirs = environment_paths("FFCV_OPENCV_LIBRARY_DIRS")
    if not include_dirs:
        for package in ("opencv5", "opencv4"):
            package_includes, package_libraries = pkg_config_paths(package)
            if package_includes:
                include_dirs = package_includes
                library_dirs = library_dirs or package_libraries
                break
    if not include_dirs:
        include_dirs = [path for path in (
            "/usr/include/opencv5", "/usr/include/opencv4",
            "/usr/local/include/opencv5", "/usr/local/include/opencv4",
        ) if Path(path).is_dir()]
    required_headers = ("opencv2/core.hpp", "opencv2/imgproc.hpp")
    if not any(all((Path(path) / header).is_file() for header in required_headers)
               for path in include_dirs):
        raise RuntimeError("Could not find OpenCV core/imgproc headers; enter the native shell or configure FFCV_OPENCV_INCLUDE_DIRS")
    return include_dirs, library_dirs


class ConfigureNativeBuild(build_ext):
    def build_extensions(self):
        includes, libraries = opencv_paths()
        jpeg_includes, jpeg_libraries = pkg_config_paths("libturbojpeg")
        for extension in self.extensions:
            if extension.name != "ffcv._libffcv":
                continue
            extension.include_dirs.extend(includes + jpeg_includes)
            extension.library_dirs.extend(libraries + jpeg_libraries)
            # The Nix build retains the exact native paths; standalone discovery still works.
            extension.runtime_library_dirs.extend(libraries + jpeg_libraries)
            extension.libraries = ["opencv_core", "opencv_imgproc", "turbojpeg", "pthread"]
            extension.extra_compile_args.append("-std=c++17")
        super().build_extensions()


setup(
    ext_modules=[Extension("ffcv._libffcv", sources=["libffcv/libffcv.cpp"])],
    cmdclass={"build_ext": ConfigureNativeBuild},
)

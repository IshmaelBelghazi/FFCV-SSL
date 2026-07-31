import os
import shlex
import subprocess
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def environment_paths(name):
    return [path for path in os.environ.get(name, "").split(os.pathsep) if path]


def pkg_config_paths(package):
    try:
        completed = subprocess.run(
            ["pkg-config", "--cflags-only-I", "--libs-only-L", package],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return [], []

    if completed.returncode != 0:
        return [], []

    tokens = shlex.split(completed.stdout)
    include_dirs = [token[2:] for token in tokens if token.startswith("-I")]
    library_dirs = [token[2:] for token in tokens if token.startswith("-L")]
    return include_dirs, library_dirs


def opencv_paths():
    include_dirs = environment_paths("FFCV_OPENCV_INCLUDE_DIRS")
    library_dirs = environment_paths("FFCV_OPENCV_LIBRARY_DIRS")

    if not include_dirs:
        for package in ("opencv5", "opencv4"):
            package_include_dirs, package_library_dirs = pkg_config_paths(package)
            if package_include_dirs:
                include_dirs = package_include_dirs
                if not library_dirs:
                    library_dirs = package_library_dirs
                break

    if not include_dirs:
        include_dirs = [
            path
            for path in (
                "/usr/include/opencv5",
                "/usr/include/opencv4",
                "/usr/local/include/opencv5",
                "/usr/local/include/opencv4",
            )
            if Path(path).is_dir()
        ]

    required_headers = ("opencv2/core.hpp", "opencv2/imgproc.hpp")
    if not any(all((Path(path) / header).is_file() for header in required_headers) for path in include_dirs):
        raise RuntimeError(
            "Could not find OpenCV core/imgproc headers. Install the OpenCV development package "
            "or set FFCV_OPENCV_INCLUDE_DIRS."
        )

    return include_dirs, library_dirs


class ConfigureNativeBuild(build_ext):
    def build_extensions(self):
        opencv_include_dirs, opencv_library_dirs = opencv_paths()
        turbojpeg_include_dirs, turbojpeg_library_dirs = pkg_config_paths("libturbojpeg")

        for extension in self.extensions:
            if extension.name != "ffcv._libffcv":
                continue
            extension.include_dirs.extend(opencv_include_dirs)
            extension.include_dirs.extend(turbojpeg_include_dirs)
            extension.library_dirs.extend(opencv_library_dirs)
            extension.library_dirs.extend(turbojpeg_library_dirs)
            extension.libraries = ["opencv_core", "opencv_imgproc", "turbojpeg", "pthread"]
            extension.extra_compile_args.append("-std=c++17")

        super().build_extensions()


libffcv = Extension("ffcv._libffcv", sources=["./libffcv/libffcv.cpp"])

setup(name='ffcv-ssl',
      version='0.0.3rc2',
      description=' FFCV-SSL: Fast Forward Computer Vision for Self-Supervised Learning ',
      url='https://github.com/facebookresearch/FFCV-SSL',
      license_files=('LICENSE',),
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      ext_modules=[libffcv],
      cmdclass={"build_ext": ConfigureNativeBuild},
      install_requires=[
            'terminaltables',
            "numpy>=2.0,<=2.3",
            "Pillow>=12.3.0",
            'pytorch_pfn_extras',
            'fastargs',
            'matplotlib',
            'scikit-learn',
            'imgcat',
            'pandas',
            'assertpy',
            'tqdm',
            'psutil',
            'webdataset',
            'submitit',
            'torchmetrics',
            'scipy'
      ]
      )

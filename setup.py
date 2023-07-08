import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


# reference: https://github.com/pybind/cmake_example
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCHATGLM_ENABLE_PYBIND=ON",
        ]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_temp, check=True)


HERE = Path(__file__).resolve().parent
version = re.search(r'__version__ = "(.*?)"', (HERE / "chatglm_cpp/__init__.py").read_text(encoding="utf-8")).group(1)

setup(
    version=version,
    packages=find_packages(),
    ext_modules=[CMakeExtension("chatglm_cpp._C")],
    cmdclass={"build_ext": CMakeBuild},
)

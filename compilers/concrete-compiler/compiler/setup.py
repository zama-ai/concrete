import os
import subprocess
import setuptools
import re

from setuptools import Extension
from setuptools.command.build_ext import build_ext


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version():
    return re.match(
        r"__version__ = \"(?P<version>.+)\"",
        read("lib/Bindings/Python/version.txt"),
    ).group("version")


class MakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def build_dir():
    path = os.environ.get("CONCRETE_COMPILER_BUILD_DIR", "build/")
    return os.path.relpath(path)


class MakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmd = ["make", "CCACHE=ON"]
        # default to dataflow_exec to ON
        dataflow_build = os.environ.get("CONCRETE_COMPILER_DATAFLOW_EXECUTION_ENABLED", "ON")
        cmd.append(f"DATAFLOW_EXECUTION_ENABLED={dataflow_build}")
        py_exec = os.environ.get("CONCRETE_COMPILER_Python3_EXECUTABLE")
        if py_exec:
            cmd.append(f"Python3_EXECUTABLE={py_exec}")
        cuda_support = os.environ.get("CONCRETE_COMPILER_CUDA_SUPPORT")
        if cuda_support:
            cmd.append(f"CUDA_SUPPORT={cuda_support}")
        cmd.append(f"BUILD_DIR={build_dir()}")
        cmd.append("python-bindings")
        subprocess.check_call(cmd)


setuptools.setup(
    name="concrete-compiler",
    version=find_version(),
    author="Zama Team",
    author_email="hello@zama.ai",
    description="Concrete Compiler",
    license="BSD-3",
    keywords="homomorphic encryption compiler",
    long_description=read("RELEASE_README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/zama-ai/concrete-compiler",
    packages=setuptools.find_namespace_packages(
        where=build_dir() + "/tools/concretelang/python_packages/concretelang_core",
        include=["concrete", "concrete.*"],
    )
    + setuptools.find_namespace_packages(
        where=build_dir() + "/tools/concretelang/python_packages/concretelang_core",
        include=["mlir", "mlir.*"],
    ),
    setup_requires=['wheel'],
    install_requires=["numpy", "PyYAML", "setuptools"],
    package_dir={"": build_dir() + "/tools/concretelang/python_packages/concretelang_core"},
    include_package_data=True,
    package_data={"": ["*.so", "*.dylib"]},
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Compilers",
        "Topic :: Security :: Cryptography",
    ],
    ext_modules=[MakeExtension("python-bindings")],
    cmdclass=dict(build_ext=MakeBuild),
    zip_safe=False,
)

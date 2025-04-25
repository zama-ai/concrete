import os
import re
import setuptools

from setuptools import Extension
from setuptools.command.build_ext import build_ext


def read(file):
    return open(os.path.join(os.path.dirname(__file__), file)).read()


def version():
    return re.match(r"__version__ = \"(?P<version>.+)\"", read("version.txt")).group("version")


def bindings_directory():
    path = os.environ.get("COMPILER_BUILD_DIRECTORY")
    if path is None or path == "":
        raise RuntimeError("COMPILER_BUILD_DIRECTORY is not set")
    return os.path.relpath(path) + "/tools/concretelang/python_packages/concretelang_core"


class MakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class MakeBuild(build_ext):
    def run(self):
        pass


def read_requirements(*filenames):
    return [
        dependency
        for filename in filenames
        for dependency in read(filename).split("\n")
        if dependency.strip() != ""
    ]

setuptools.setup(

    name="concrete-python",
    description="A state-of-the-art homomorphic encryption framework",

    version=version(),
    license="BSD-3-Clause",

    author="Zama",
    author_email="hello@zama.ai",

    url="https://github.com/zama-ai/concrete/tree/main/frontends/concrete-python",
    keywords=[
        "fhe",
        "homomorphic",
        "encryption",
        "tfhe",
        "privacy",
        "security",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Compilers",
    ],

    long_description=read("../../README.md"),
    long_description_content_type="text/markdown",

    python_requires=">=3.9",
    setup_requires=["wheel"],
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements.dev.txt", "requirements.extra-full.txt"),
        "full": read_requirements("requirements.extra-full.txt"),
    },
    package_dir={
        "concrete.fhe": "./concrete/fhe",
        "": bindings_directory(),
    },
    packages=setuptools.find_namespace_packages(
        where=".",
        include=["concrete", "concrete.*"]
    ) + setuptools.find_namespace_packages(
        where=".",
        include=["concrete.fhe", "concrete.fhe.*"],
    ) + setuptools.find_namespace_packages(
        where=bindings_directory(),
        include=["concrete.compiler", "concrete.compiler.*"],
    ) + setuptools.find_namespace_packages(
        where=bindings_directory(),
        include=["concrete.lang", "concrete.lang.*"],
    ) + setuptools.find_namespace_packages(
        where=bindings_directory(),
        include=["mlir", "mlir.*"],
    ),

    include_package_data=True,
    package_data={"": ["*.so", "*.dylib"]},

    ext_modules=[MakeExtension("python-bindings")],
    cmdclass=dict(build_ext=MakeBuild),
    zip_safe=False,

)

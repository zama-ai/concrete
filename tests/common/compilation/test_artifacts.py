"""Test file for compilation artifacts"""

import tempfile
from pathlib import Path

from hdk.common.compilation import CompilationArtifacts
from hdk.common.data_types.integers import UnsignedInteger
from hdk.common.values import EncryptedScalar
from hdk.hnumpy.compile import compile_numpy_function


def test_artifacts_export():
    """Test function to check exporting compilation artifacts"""

    def function(x):
        return x + 42

    with tempfile.TemporaryDirectory() as tmp:
        output_directory = Path(tmp)
        artifacts = CompilationArtifacts(output_directory)

        compile_numpy_function(
            function,
            {"x": EncryptedScalar(UnsignedInteger(7))},
            iter([(0,), (1,), (2,)]),
            compilation_artifacts=artifacts,
        )

        artifacts.export()

        assert output_directory.joinpath("environment.txt").exists()
        assert output_directory.joinpath("requirements.txt").exists()

        assert output_directory.joinpath("function.txt").exists()
        assert output_directory.joinpath("parameters.txt").exists()

        assert output_directory.joinpath("1.initial.graph.txt").exists()
        assert output_directory.joinpath("1.initial.graph.png").exists()

        assert output_directory.joinpath("2.final.graph.txt").exists()
        assert output_directory.joinpath("2.final.graph.png").exists()

        assert output_directory.joinpath("bounds.txt").exists()
        assert output_directory.joinpath("mlir.txt").exists()

        # format of those files might change in the future
        # so it is sufficient to test their existance

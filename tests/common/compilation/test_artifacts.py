"""Test file for compilation artifacts"""

import tempfile
from pathlib import Path

from hdk.common.compilation import CompilationArtifacts
from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import EncryptedValue
from hdk.hnumpy.compile import compile_numpy_function


def test_artifacts_export():
    """Test function to check exporting compilation artifacts"""

    def function(x):
        return x + 42

    artifacts = CompilationArtifacts()
    compile_numpy_function(
        function,
        {"x": EncryptedValue(Integer(7, True))},
        iter([(-2,), (-1,), (0,), (1,), (2,)]),
        artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_directory = Path(tmp)
        artifacts.export(output_directory)

        assert output_directory.joinpath("environment.txt").exists()
        assert output_directory.joinpath("requirements.txt").exists()
        assert output_directory.joinpath("graph.txt").exists()
        assert output_directory.joinpath("bounds.txt").exists()

        # format of those files might change in the future
        # so it is sufficient to test their existance

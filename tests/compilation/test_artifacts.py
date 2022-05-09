"""
Tests of `DebugArtifacts` class.
"""

import tempfile
from pathlib import Path

from concrete.numpy.compilation import DebugArtifacts, compiler


def test_artifacts_export(helpers):
    """
    Test `export` method of `DebugArtifacts` class.
    """

    with tempfile.TemporaryDirectory() as path:
        tmpdir = Path(path)

        configuration = helpers.configuration()
        artifacts = DebugArtifacts(tmpdir)

        @compiler({"x": "encrypted"})
        def f(x):
            return x + 10

        inputset = range(100)
        f.compile(inputset, configuration, artifacts)

        artifacts.export()

        assert (tmpdir / "environment.txt").exists()
        assert (tmpdir / "requirements.txt").exists()

        assert (tmpdir / "function.txt").exists()
        assert (tmpdir / "parameters.txt").exists()

        assert (tmpdir / "1.initial.graph.txt").exists()
        assert (tmpdir / "1.initial.graph.png").exists()

        assert (tmpdir / "2.final.graph.txt").exists()
        assert (tmpdir / "2.final.graph.png").exists()

        assert (tmpdir / "bounds.txt").exists()
        assert (tmpdir / "mlir.txt").exists()
        assert (tmpdir / "client_parameters.json").exists()

        artifacts.export()

        assert (tmpdir / "environment.txt").exists()
        assert (tmpdir / "requirements.txt").exists()

        assert (tmpdir / "function.txt").exists()
        assert (tmpdir / "parameters.txt").exists()

        assert (tmpdir / "1.initial.graph.txt").exists()
        assert (tmpdir / "1.initial.graph.png").exists()

        assert (tmpdir / "2.final.graph.txt").exists()
        assert (tmpdir / "2.final.graph.png").exists()

        assert (tmpdir / "bounds.txt").exists()
        assert (tmpdir / "mlir.txt").exists()
        assert (tmpdir / "client_parameters.json").exists()

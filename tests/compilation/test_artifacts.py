"""
Tests of `CompilationArtifacts` class.
"""

import tempfile
from pathlib import Path

from concrete.numpy.compilation import CompilationArtifacts, compiler


def test_artifacts_export(helpers):
    """
    Test `export` method of `Compilation` class.
    """

    with tempfile.TemporaryDirectory() as path:
        tmpdir = Path(path)

        configuration = helpers.configuration()
        artifacts = CompilationArtifacts(tmpdir)

        @compiler({"x": "encrypted"}, configuration=configuration, artifacts=artifacts)
        def f(x):
            return x + 10

        inputset = range(100)
        f.compile(inputset)

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

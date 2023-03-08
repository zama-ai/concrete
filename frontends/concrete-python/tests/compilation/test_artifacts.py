"""
Tests of `DebugArtifacts` class.
"""

import tempfile
from pathlib import Path

import numpy as np

from concrete.numpy import DebugArtifacts, compiler


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
            a = ((np.sin(x) ** 2) + (np.cos(x) ** 2)).astype(np.int64)
            b = np.where(x < 5, x * 10, x + 10)
            return a + b

        inputset = range(10)
        f.compile(inputset, configuration, artifacts)

        artifacts.export()

        assert (tmpdir / "environment.txt").exists()
        assert (tmpdir / "requirements.txt").exists()

        assert (tmpdir / "function.txt").exists()
        assert (tmpdir / "parameters.txt").exists()

        assert (tmpdir / "1.initial.graph.txt").exists()
        assert (tmpdir / "2.after-fusing.graph.txt").exists()
        assert (tmpdir / "3.after-fusing.graph.txt").exists()
        assert (tmpdir / "4.final.graph.txt").exists()

        assert (tmpdir / "mlir.txt").exists()
        assert (tmpdir / "client_parameters.json").exists()

        artifacts.export()

        assert (tmpdir / "environment.txt").exists()
        assert (tmpdir / "requirements.txt").exists()

        assert (tmpdir / "function.txt").exists()
        assert (tmpdir / "parameters.txt").exists()

        assert (tmpdir / "1.initial.graph.txt").exists()
        assert (tmpdir / "2.after-fusing.graph.txt").exists()
        assert (tmpdir / "3.after-fusing.graph.txt").exists()
        assert (tmpdir / "4.final.graph.txt").exists()

        assert (tmpdir / "mlir.txt").exists()
        assert (tmpdir / "client_parameters.json").exists()

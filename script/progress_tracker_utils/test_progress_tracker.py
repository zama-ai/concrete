"""Test file for progress tracker"""

from typing import List

import measure
import pytest


class Args:
    """Class to mimic the command line arguments that can be passed to measurement script."""

    base: str
    files_to_benchmark: List[str]
    samples: int
    keep: bool
    check: bool

    def __init__(self, files_to_benchmark: List[str]):
        self.base = "script/progress_tracker_utils/test_scripts"
        self.files_to_benchmark = files_to_benchmark
        self.samples = 30
        self.keep = False
        self.check = True


def test_alert_on_undefined_metric():
    """Test function for alert directive on unefined metric"""

    file = "script/progress_tracker_utils/test_scripts/alert_on_undefined_metric.py"
    args = Args([file])

    with pytest.raises(SyntaxError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == (
        f"An alert is using an undefined metric `Accuracy (%)` (at line 7 of {file})"
    )


def test_alert_invalid_comparison_constant():
    """Test function for alert directive with invalid constant"""

    file = "script/progress_tracker_utils/test_scripts/alert_invalid_comparison_constant.py"
    args = Args([file])

    with pytest.raises(SyntaxError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == (
        f"An alert is not using a constant floating point for comparison "
        f'(it uses `"abc"` at line 7 of {file})'
    )


def test_alert_invalid_comparison_operator():
    """Test function for alert directive that use invalid comparison operator"""

    file = "script/progress_tracker_utils/test_scripts/alert_invalid_comparison_operator.py"
    args = Args([file])

    with pytest.raises(SyntaxError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == (
        f"An alert is not using any of the supported comparisons ==, !=, <=, >=, <, > "
        f"(at line 7 of {file})"
    )


def test_measure_end_before_start():
    """Test function for measure end directive before measure directive"""

    file = "script/progress_tracker_utils/test_scripts/measure_end_before_start.py"
    args = Args([file])

    with pytest.raises(SyntaxError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == (
        f"Measurements cannot end before they are defined (at line 3 of {file})"
    )


def test_measure_invalid_indentation():
    """Test function for invalid indentation of measure directives"""

    file = "script/progress_tracker_utils/test_scripts/measure_invalid_indentation.py"
    args = Args([file])

    with pytest.raises(SyntaxError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == (
        f"Measurements should finish with the same indentation as they are defined "
        f"(at lines 4 and 6 of {file})"
    )


def test_measure_nested():
    """Test function for nested measure directives"""

    file = "script/progress_tracker_utils/test_scripts/measure_nested.py"
    args = Args([file])

    with pytest.raises(SyntaxError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == (
        f"Nested measurements are not supported (at lines 3 and 7 of {file})"
    )


def test_measure_unfinished():
    """Test function for measure directives without a measure end directive"""

    file = "script/progress_tracker_utils/test_scripts/measure_unfinished.py"
    args = Args([file])

    with pytest.raises(SyntaxError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == (
        f"Unfinished measurements are not supported (at line 3 of {file})"
    )


def test_two_targets_with_the_same_name():
    """Test function for target name collisions"""

    file1 = "script/progress_tracker_utils/test_scripts/two_targets_with_the_same_name.1.py"
    file2 = "script/progress_tracker_utils/test_scripts/two_targets_with_the_same_name.2.py"
    args = Args([file1, file2])

    with pytest.raises(RuntimeError) as excinfo:
        measure.main(args)

    assert str(excinfo.value) == "Target `X` is already registered"

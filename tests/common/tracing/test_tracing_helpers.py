"""Test file for HDK's common tracing helpers"""

from typing import Any, Dict

import pytest

from hdk.common.tracing.tracing_helpers import prepare_function_parameters


@pytest.mark.parametrize(
    "function,function_parameters,ref_dict",
    [
        pytest.param(lambda x: None, {}, {}, id="Missing x", marks=pytest.mark.xfail(strict=True)),
        pytest.param(lambda x: None, {"x": None}, {"x": None}, id="Only x"),
        pytest.param(
            lambda x: None, {"x": None, "y": None}, {"x": None}, id="Additional y filtered"
        ),
    ],
)
def test_prepare_function_parameters(
    function, function_parameters: Dict[str, Any], ref_dict: Dict[str, Any]
):
    """Test prepare_function_parameters"""
    prepared_dict = prepare_function_parameters(function, function_parameters)

    assert prepared_dict == ref_dict

import pytest
from concrete.compiler import (
    ClientParameters,
    ClientSupport,
    CompilationOptions,
    JITCompilationResult,
    JITLambda,
    JITSupport,
    KeySetCache,
    KeySet,
    LambdaArgument,
    LibraryCompilationResult,
    LibraryLambda,
    LibrarySupport,
    PublicArguments,
    PublicResult,
)


@pytest.mark.parametrize("garbage", ["string here", 23, None])
@pytest.mark.parametrize(
    "WrapperClass",
    [
        pytest.param(ClientParameters, id="ClientParameters"),
        pytest.param(ClientSupport, id="ClientSupport"),
        pytest.param(CompilationOptions, id="CompilationOptions"),
        pytest.param(JITCompilationResult, id="JITCompilationResult"),
        pytest.param(JITLambda, id="JITLambda"),
        pytest.param(JITSupport, id="JITSupport"),
        pytest.param(KeySetCache, id="KeySetCache"),
        pytest.param(KeySet, id="KeySet"),
        pytest.param(LambdaArgument, id="LambdaArgument"),
        pytest.param(LibraryCompilationResult, id="LibraryCompilationResult"),
        pytest.param(LibraryLambda, id="LibraryLambda"),
        pytest.param(LibrarySupport, id="LibrarySupport"),
        pytest.param(PublicArguments, id="PublicArguments"),
        pytest.param(PublicResult, id="PublicResult"),
    ],
)
def test_invalid_wrapping(WrapperClass, garbage):
    with pytest.raises(
        TypeError,
        match=f"\.* must be of type _{WrapperClass.__name__}, not {type(garbage)}",
    ):
        WrapperClass(garbage)

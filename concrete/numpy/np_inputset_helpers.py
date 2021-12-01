"""Helpers for numpy inputset related functionality."""

import random
from typing import Any, Dict, Iterable, Tuple, Union

import numpy

from ..common.compilation import CompilationConfiguration
from ..common.data_types import Float, Integer
from ..common.values import BaseValue, TensorValue


def _generate_random_integer_scalar(dtype: Integer) -> int:
    """Generate a random integer scalar.

    Args:
        dtype (Integer): the data type to extract bounds

    Returns:
        int: a random value within the range [dtype.min_value(), dtype.max_value()]
    """

    return random.randint(dtype.min_value(), dtype.max_value())


def _generate_random_integer_tensor(dtype: Integer, shape: Tuple[int, ...]) -> numpy.ndarray:
    """Generate a random integer tensor.

    Args:
        dtype (Integer): the data type to extract bounds
        shape (Tuple[int, ...]): the shape of the generated tensor

    Returns:
        numpy.ndarray: a random array of the specified shape where each value of it
            is within the range [dtype.min_value(), dtype.max_value()]
    """

    return numpy.random.randint(
        dtype.min_value(),
        dtype.max_value() + 1,
        size=shape,
        dtype=numpy.int64 if dtype.is_signed else numpy.uint64,  # type: ignore
    )


def _generate_random_float_scalar() -> float:
    """Generate a random float scalar.

    Returns:
        float: a random value within the range [0, 1)
    """

    return random.random()


def _generate_random_float_tensor(dtype: Float, shape: Tuple[int, ...]) -> numpy.ndarray:
    """Generate a random float tensor.

    Args:
        dtype (Integer): the data type to extract resulting numpy data type
        shape (Tuple[int, ...]): the shape of the generated tensor

    Returns:
        numpy.ndarray: a random array of the specified shape where each value of it
            is within the range [0, 1)
    """

    result = numpy.random.rand(*shape)
    return result.astype(numpy.float32 if dtype.bit_width == 32 else numpy.float64)


def _generate_random_inputset(
    function_parameters: Dict[str, BaseValue],
    compilation_configuration: CompilationConfiguration,
) -> Union[Iterable[Any], Iterable[Tuple[Any, ...]]]:
    """Generate a random inputset from function parameters.

    Using this function is not a good practice since the randomly generated inputset
    might not reflect real world data. We have it to speed up our development workflow
    and we also don't use it in any of our tests, benchmarks, or examples.

    Args:
        function_parameters (Dict[str, BaseValue]): the function parameters
            to extract data types and shapes
        compilation_configuration (CompilationConfiguration): the compilation configuration
            to extract the sample size of the resulting inputset

    Raises:
        ValueError: if the provided function arguments cannot be used for random inputset generation

    Returns:
        Union[Iterable[Any], Iterable[Tuple[Any, ...]]]: the inputset
    """

    inputset = []
    for _ in range(compilation_configuration.random_inputset_samples):
        sample = []
        for parameter in function_parameters.values():
            if not isinstance(parameter, TensorValue):
                raise ValueError(f"Random inputset cannot be generated for {parameter} parameters")

            if isinstance(parameter.dtype, Integer):
                sample.append(
                    _generate_random_integer_scalar(parameter.dtype)
                    if parameter.is_scalar
                    else _generate_random_integer_tensor(parameter.dtype, parameter.shape)
                )
            elif isinstance(parameter.dtype, Float):
                sample.append(
                    _generate_random_float_scalar()
                    if parameter.is_scalar
                    else _generate_random_float_tensor(parameter.dtype, parameter.shape)
                )
            else:
                raise ValueError(
                    f"Random inputset cannot be generated "
                    f"for parameters of type {parameter.dtype}"
                )
        inputset.append(tuple(sample) if len(sample) > 1 else sample[0])
    return inputset


def _check_special_inputset_availability(
    inputset: str,
    compilation_configuration: CompilationConfiguration,
):
    """Check special inputset is valid and is available.

    This function makes sure the provided special inputset is valid and can be used with the
    provided compilation configuration.

    Currently, the only special inputset is "random" but this can be extended in the future.

    Args:
        inputset (str): the special inputset to check
        compilation_configuration (CompilationConfiguration): the compilation configuration
            to check the availability of the provided special inputset

    Raises:
        ValueError: if the provided special inputset is not valid
        RuntimeError: if the provided special inputset is not available

    Returns:
        None
    """

    if inputset != "random":
        raise ValueError(
            f"inputset can only be an iterable of tuples or the string 'random' "
            f"but you specified '{inputset}' for it"
        )

    if not compilation_configuration.enable_unsafe_features:
        raise RuntimeError(
            "Random inputset generation is an unsafe feature and should not be used "
            "if you don't know what you are doing"
        )

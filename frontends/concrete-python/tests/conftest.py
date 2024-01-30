"""
Configuration of `pytest`.
"""

import json
import os
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pytest

import tests
from concrete import fhe

tests_directory = os.path.dirname(tests.__file__)


INSECURE_KEY_CACHE_LOCATION = None
USE_MULTI_PRECISION = False
OPTIMIZATION_STRATEGY = fhe.ParameterSelectionStrategy.MONO
USE_GPU = False


def pytest_addoption(parser):
    """
    Add CLI options.
    """

    parser.addoption(
        "--global-coverage",
        type=str,
        default=None,
        action="store",
        help="JSON file to dump pytest-cov terminal report.",
    )
    parser.addoption(
        "--key-cache",
        type=str,
        default=None,
        action="store",
        help="Specify the location of the key cache",
    )
    parser.addoption(
        "--precision",
        type=str,
        default=None,
        action="store",
        help="Which precision strategy to use in execution tests (single or multi)",
    )
    parser.addoption(
        "--strategy",
        type=str,
        default=None,
        action="store",
        help="Which optimization strategy to use in execution tests (v0, mono or multi)",
    )
    parser.addoption("--use_gpu", action="store_true")


def pytest_sessionstart(session):
    """
    Initialize insecure key cache.
    """
    # pylint: disable=global-statement
    global INSECURE_KEY_CACHE_LOCATION
    global USE_MULTI_PRECISION
    global OPTIMIZATION_STRATEGY
    global USE_GPU
    # pylint: enable=global-statement

    key_cache_location = session.config.getoption("--key-cache", default=None)
    if key_cache_location is not None and key_cache_location != "":
        if key_cache_location.lower() == "disable":
            key_cache_location = None
        else:
            key_cache_location = Path(key_cache_location).expanduser().resolve()
    else:
        key_cache_location = Path.home().resolve() / ".cache" / "concrete-python" / "pytest"

    if key_cache_location:
        key_cache_location.mkdir(parents=True, exist_ok=True)
        print(f"INSECURE_KEY_CACHE_LOCATION={str(key_cache_location)}")

        INSECURE_KEY_CACHE_LOCATION = str(key_cache_location)

    precision = session.config.getoption("--precision", default="single")
    USE_MULTI_PRECISION = precision == "multi"

    strategy = session.config.getoption("--strategy", default="mono")
    if strategy == "v0":
        OPTIMIZATION_STRATEGY = fhe.ParameterSelectionStrategy.V0
    elif strategy == "multi":
        OPTIMIZATION_STRATEGY = fhe.ParameterSelectionStrategy.MULTI
    else:
        OPTIMIZATION_STRATEGY = fhe.ParameterSelectionStrategy.MONO

    USE_GPU = session.config.getoption("--use_gpu", default=False)


def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """
    Save global coverage info after testing is finished.
    """

    # Hacked together from the source code, they don't have an option to export to file,
    # and it's too much work to get a PR in for such a little thing.
    # https://github.com/pytest-dev/pytest-cov/blob/ec344d8adf2d78238d8f07cb20ed2463d7536970/src/pytest_cov/plugin.py#L329
    if session.config.pluginmanager.hasplugin("_cov"):
        global_coverage_option = session.config.getoption("--global-coverage", default=None)
        if global_coverage_option is not None:
            coverage_plugin = session.config.pluginmanager.getplugin("_cov")
            coverage_txt = coverage_plugin.cov_report.getvalue()

            coverage_status = 0
            if (
                coverage_plugin.options.cov_fail_under is not None
                and coverage_plugin.options.cov_fail_under > 0
                and coverage_plugin.cov_total < coverage_plugin.options.cov_fail_under
            ):
                coverage_status = 1

            global_coverage_file_path = Path(global_coverage_option).resolve()
            with open(global_coverage_file_path, "w", encoding="utf-8") as f:
                json.dump({"exit_code": coverage_status, "content": coverage_txt}, f)


class Helpers:
    """
    Helpers class, which provides various helpers to tests.
    """

    @staticmethod
    def configuration() -> fhe.Configuration:
        """
        Get the test configuration to use during testing.

        Returns:
            fhe.Configuration:
                test configuration
        """

        return fhe.Configuration(
            dump_artifacts_on_unexpected_failures=False,
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            loop_parallelize=True,
            dataflow_parallelize=False,
            auto_parallelize=False,
            insecure_key_cache_location=INSECURE_KEY_CACHE_LOCATION,
            global_p_error=(1 / 10_000),
            single_precision=(not USE_MULTI_PRECISION),
            parameter_selection_strategy=OPTIMIZATION_STRATEGY,
            use_gpu=USE_GPU,
            compress_evaluation_keys=True,
        )

    @staticmethod
    def generate_encryption_statuses(parameters: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate parameter encryption statuses accoring to a parameter specification.

        Args:
            parameters (Dict[str, Dict[str, Any]]):
                parameter specification to use

                e.g.,

                {
                    "param1": {"range": [0, 10], "status": "clear"},
                    "param2": {"range": [3, 30], "status": "encrypted", "shape": (3,)},
                }

        Returns:
            Dict[str, str]:
                parameter encryption statuses
                generated according to the given parameter specification
        """

        return {
            parameter: details["status"] if "status" in details else "encrypted"
            for parameter, details in parameters.items()
        }

    @staticmethod
    def generate_inputset(
        parameters: Dict[str, Dict[str, Any]],
        size: int = 128,
    ) -> List[Union[Tuple[Union[int, np.ndarray], ...], Union[int, np.ndarray]]]:
        """
        Generate a random inputset of desired size accoring to a parameter specification.

        Args:
            parameters (Dict[str, Dict[str, Any]]):
                parameter specification to use

                e.g.,

                {
                    "param1": {"range": [0, 10], "status": "clear"},
                    "param2": {"range": [3, 30], "status": "encrypted", "shape": (3,)},
                }

            size (int):
                size of the resulting inputset

        Returns:
            List[Union[Tuple[Union[int, np.ndarray], ...], Union[int, np.ndarray]]]:
                random inputset of desired size
                generated according to the given parameter specification
        """

        inputset = []

        for _ in range(size):
            sample = Helpers.generate_sample(parameters)
            inputset.append(tuple(sample) if len(sample) > 1 else sample[0])

        return inputset

    @staticmethod
    def generate_sample(parameters: Dict[str, Dict[str, Any]]) -> List[Union[int, np.ndarray]]:
        """
        Generate a random sample accoring to a parameter specification.

        Args:
            parameters (Dict[str, Dict[str, Any]]):
                parameter specification to use

                e.g.,

                {
                    "param1": {"range": [0, 10], "status": "clear"},
                    "param2": {"range": [3, 30], "status": "encrypted", "shape": (3,)},
                }

        Returns:
            List[Union[int, np.ndarray]]:
                random sample
                generated according to the given parameter specification
        """

        sample = []

        for description in parameters.values():
            minimum, maximum = description.get("range", [0, (2**16) - 1])

            if "shape" in description:
                shape = description["shape"]
                sample.append(np.random.randint(minimum, maximum + 1, size=shape, dtype=np.int64))
            else:
                sample.append(np.int64(random.randint(minimum, maximum)))

        return sample

    @staticmethod
    def check_execution(
        circuit: fhe.Circuit,
        function: Callable,
        sample: Union[Any, List[Any]],
        retries: int = 1,
        only_simulation: bool = False,
    ):
        """
        Assert that `circuit` behaves the same as `function` on `sample`.

        Args:
            circuit (fhe.Circuit):
                compiled circuit

            function (Callable):
                original function

            sample (List[Any]):
                inputs

            retries (int, default = 1):
                number of times to retry (for probabilistic execution)

            only_simulation (bool, default = False):
                whether to just check simulation but not execution
        """
        if not isinstance(sample, list):
            sample = [sample]

        def sanitize(values):
            if not isinstance(values, tuple):
                values = (values,)

            result = []
            for value in values:
                if isinstance(value, (bool, np.bool_)):
                    value = int(value)
                elif isinstance(value, np.ndarray) and value.dtype == np.bool_:
                    value = value.astype(np.int64)

                result.append(value)

            return tuple(result)

        if not only_simulation:
            for i in range(retries):
                expected = sanitize(function(*deepcopy(sample)))
                actual = sanitize(circuit.encrypt_run_decrypt(*deepcopy(sample)))

                if all(np.array_equal(e, a) for e, a in zip(expected, actual)):
                    break

                warnings.warn(
                    UserWarning(f"Test fail ({i+1}/{retries}), regenerate keyset and retry"),
                    stacklevel=2,
                )
                circuit.keygen(force=True, seed=i + 1)

                if i == retries - 1:
                    message = f"""

    Expected Output
    ===============
    {expected}

    Actual Output
    =============
    {actual}

                        """
                    raise AssertionError(message)

        circuit.enable_fhe_simulation()

        # Skip simulation for GPU
        if circuit.configuration.use_gpu:
            return

        for i in range(retries):
            expected = sanitize(function(*deepcopy(sample)))
            actual = sanitize(circuit.simulate(*deepcopy(sample)))

            if all(np.array_equal(e, a) for e, a in zip(expected, actual)):
                break

            if i == retries - 1:
                message = f"""

Expected Output During Simulation
=================================
{expected}

Actual Output During Simulation
===============================
{actual}

                """
                raise AssertionError(message)

    @staticmethod
    def check_composition(
        circuit: fhe.Circuit, function: Callable, sample: Union[Any, List[Any]], composed: int
    ):
        """
        Assert that `circuit` behaves the same as `function` on `sample` when composed.

        Args:
            circuit (fhe.Circuit):
                compiled circuit

            function (Callable):
                original function

            sample (List[Any]):
                inputs

            composed (int):
                number of times to compose the function (call sequentially with inputs as outputs)
        """

        if not isinstance(sample, list):
            sample = [sample]

        def sanitize(values):
            if not isinstance(values, tuple):
                values = (values,)

            result = []
            for value in values:
                if isinstance(value, (bool, np.bool_)):
                    value = int(value)
                elif isinstance(value, np.ndarray) and value.dtype == np.bool_:
                    value = value.astype(np.int64)

                result.append(value)

            return tuple(result)

        def compute_expected(sample):
            for _ in range(composed):
                sample = function(*sample)
                if not isinstance(sample, tuple):
                    sample = (sample,)
            return sample

        def compute_actual(sample):
            inp = circuit.encrypt(*sample)
            for _ in range(composed):
                inp = circuit.run(inp)
            out = circuit.decrypt(inp)
            return out

        expected = sanitize(compute_expected(sample))
        actual = sanitize(compute_actual(sample))

        if not all(np.array_equal(e, a) for e, a in zip(expected, actual)):
            message = f"""

    Expected Output
    ===============
    {expected}

    Actual Output
    =============
    {actual}
            """
            raise AssertionError(message)

    @staticmethod
    def check_str(expected: str, actual: str):
        """
        Assert that `circuit` is behaves the same as `function` on `sample`.

        Args:
            expected (str):
                expected str

            actual (str):
                actual str
        """

        # remove error line information
        # there are explicit tests to make sure the line information is correct
        # however, it would have been very hard to keep the other tests up to date

        actual = "\n".join(
            line for line in actual.splitlines() if not line.strip().startswith(tests_directory)
        )

        assert (
            actual.strip() == expected.strip()
        ), f"""

Expected Output
===============
{expected}

Actual Output
=============
{actual}

            """


@pytest.fixture
def helpers():
    """
    Fixture that provides `Helpers` class to tests.
    """

    return Helpers

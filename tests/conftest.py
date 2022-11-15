"""
Configuration of `pytest`.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pytest

import concrete.numpy as cnp
import tests

tests_directory = os.path.dirname(tests.__file__)


INSECURE_KEY_CACHE_LOCATION = None


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


def pytest_sessionstart(session):
    """
    Initialize insecure key cache.
    """
    # pylint: disable=global-statement
    global INSECURE_KEY_CACHE_LOCATION
    # pylint: enable=global-statement

    key_cache_location = session.config.getoption("--key-cache", default=None)
    if key_cache_location is not None:
        if key_cache_location.lower() == "disable":
            key_cache_location = None
        else:
            key_cache_location = Path(key_cache_location).expanduser().resolve()
    else:
        key_cache_location = Path.home().resolve() / ".cache" / "concrete-numpy" / "pytest"

    if key_cache_location:
        key_cache_location.mkdir(parents=True, exist_ok=True)
        print(f"INSECURE_KEY_CACHE_LOCATION={str(key_cache_location)}")

        INSECURE_KEY_CACHE_LOCATION = str(key_cache_location)


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
    def configuration() -> cnp.Configuration:
        """
        Get the test configuration to use during testing.

        Returns:
            cnp.Configuration:
                test configuration
        """

        return cnp.Configuration(
            dump_artifacts_on_unexpected_failures=False,
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            loop_parallelize=True,
            dataflow_parallelize=False,
            auto_parallelize=False,
            jit=True,
            insecure_key_cache_location=INSECURE_KEY_CACHE_LOCATION,
            global_p_error=(1 / 10_000),
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
        circuit: cnp.Circuit,
        function: Callable,
        sample: Union[Any, List[Any]],
        retries: int = 1,
    ):
        """
        Assert that `circuit` is behaves the same as `function` on `sample`.

        Args:
            circuit (cnp.Circuit):
                compiled circuit

            function (Callable):
                original function

            sample (List[Any]):
                inputs

            retries (int):
                number of times to retry (for probabilistic execution)
        """

        if not isinstance(sample, list):
            sample = [sample]

        for i in range(retries):
            expected = function(*sample)
            actual = circuit.encrypt_run_decrypt(*sample)

            if not isinstance(expected, tuple):
                expected = (expected,)
            if not isinstance(actual, tuple):
                actual = (actual,)

            if all(np.array_equal(e, a) for e, a in zip(expected, actual)):
                break

            if i == retries - 1:
                raise AssertionError(
                    f"""

Expected Output
===============
{expected}

Actual Output
=============
{actual}

                    """
                )

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

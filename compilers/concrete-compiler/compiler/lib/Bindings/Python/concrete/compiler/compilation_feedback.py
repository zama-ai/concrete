#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

"""Compilation feedback."""

import re
from typing import Dict, Set

# pylint: disable=no-name-in-module,import-error,too-many-instance-attributes
from mlir._mlir_libs._concretelang._compiler import (
    CircuitCompilationFeedback,
    KeyType,
    PrimitiveOperation,
    ProgramInfo,
)

# pylint: enable=no-name-in-module,import-error

# matches (@tag, separator( | ), filename)
REGEX_LOCATION = re.compile(r"loc\(\"(@[\w\.]+)?( \| )?(.+)\"")


def tag_from_location(location):
    """
    Extract tag of the operation from its location.
    """

    match = REGEX_LOCATION.match(location)
    if match is not None:
        tag, _, _ = match.groups()
        # remove the @
        tag = tag[1:] if tag else ""
    else:
        tag = ""
    return tag


class MoreCircuitCompilationFeedback:
    """
    Helper class for compilation feedback.
    """

    @staticmethod
    def count(
        circuit_feedback: CircuitCompilationFeedback,
        *,
        operations: Set[PrimitiveOperation],
    ) -> int:
        """
        Count the amount of specified operations in the program.

        Args:
            operations (Set[PrimitiveOperation]):
                set of operations used to filter the statistics

        Returns:
            int:
                number of specified operations in the program
        """

        return sum(
            statistic.count
            for statistic in circuit_feedback.statistics
            if statistic.operation in operations
        )

    @staticmethod
    def count_per_parameter(
        circuit_feedback: CircuitCompilationFeedback,
        *,
        operations: Set[PrimitiveOperation],
        key_types: Set[KeyType],
        program_info: ProgramInfo,
    ) -> Dict["Parameter", int]:
        """
        Count the amount of specified operations in the program and group by parameters.

        Args:
            operations (Set[PrimitiveOperation]):
                set of operations used to filter the statistics

            key_types (Set[KeyType]):
                set of key types used to filter the statistics

            program_info (ProgramInfo):
                program info required for grouping by parameters

        Returns:
            Dict[Parameter, int]:
                number of specified operations per parameter in the program
        """

        result = {}
        for statistic in circuit_feedback.statistics:
            if statistic.operation not in operations:
                continue

            for key_type, key_index in statistic.keys:
                if key_type not in key_types:
                    continue

                if key_type == KeyType.SECRET:
                    parameter = program_info.get_keyset_info().secret_keys()[key_index]
                elif key_type == KeyType.BOOTSTRAP:
                    parameter = program_info.get_keyset_info().bootstrap_keys()[
                        key_index
                    ]
                elif key_type == KeyType.KEY_SWITCH:
                    parameter = program_info.get_keyset_info().keyswitch_keys()[
                        key_index
                    ]
                elif key_type == KeyType.PACKING_KEY_SWITCH:
                    parameter = program_info.get_keyset_info().packing_keyswitch_keys()[
                        key_index
                    ]
                else:
                    assert False
                if parameter not in result:
                    result[parameter] = 0
                result[parameter] += statistic.count

        return result

    @staticmethod
    def count_per_tag(
        circuit_feedback: CircuitCompilationFeedback,
        *,
        operations: Set[PrimitiveOperation],
    ) -> Dict[str, int]:
        """
        Count the amount of specified operations in the program and group by tags.

        Args:
            operations (Set[PrimitiveOperation]):
                set of operations used to filter the statistics

        Returns:
            Dict[str, int]:
                number of specified operations per tag in the program
        """

        result = {}
        for statistic in circuit_feedback.statistics:
            if statistic.operation not in operations:
                continue

            tag = tag_from_location(statistic.location)

            tag_components = tag.split(".")
            for i in range(1, len(tag_components) + 1):
                current_tag = ".".join(tag_components[0:i])
                if current_tag == "":
                    continue

                if current_tag not in result:
                    result[current_tag] = 0

                result[current_tag] += statistic.count

        return result

    @staticmethod
    # pylint: disable=too-many-branches
    def count_per_tag_per_parameter(
        circuit_feedback: CircuitCompilationFeedback,
        *,
        operations: Set[PrimitiveOperation],
        key_types: Set[KeyType],
        program_info: ProgramInfo,
    ) -> Dict[str, Dict["Parameter", int]]:
        """
        Count the amount of specified operations in the program and group by tags and parameters.

        Args:
            operations (Set[PrimitiveOperation]):
                set of operations used to filter the statistics

            key_types (Set[KeyType]):
                set of key types used to filter the statistics

            program_info (ProgramInfo):
                program info required for grouping by parameters

        Returns:
            Dict[str, Dict[Parameter, int]]:
                number of specified operations per tag per parameter in the program
        """

        result: Dict[str, Dict[int, int]] = {}
        for statistic in circuit_feedback.statistics:
            if statistic.operation not in operations:
                continue

            tag = tag_from_location(statistic.location)

            tag_components = tag.split(".")
            for i in range(1, len(tag_components) + 1):
                current_tag = ".".join(tag_components[0:i])
                if current_tag == "":
                    continue

                if current_tag not in result:
                    result[current_tag] = {}

                for key_type, key_index in statistic.keys:
                    if key_type not in key_types:
                        continue

                    if key_type == KeyType.SECRET:
                        parameter = program_info.get_keyset_info().secret_keys()[
                            key_index
                        ]
                    elif key_type == KeyType.BOOTSTRAP:
                        parameter = program_info.get_keyset_info().bootstrap_keys()[
                            key_index
                        ]
                    elif key_type == KeyType.KEY_SWITCH:
                        parameter = program_info.get_keyset_info().keyswitch_keys()[
                            key_index
                        ]
                    elif key_type == KeyType.PACKING_KEY_SWITCH:
                        parameter = (
                            program_info.get_keyset_info().packing_keyswitch_keys()[
                                key_index
                            ]
                        )
                    else:
                        assert False

                    if parameter not in result[current_tag]:
                        result[current_tag][parameter] = 0
                    result[current_tag][parameter] += statistic.count

        return result

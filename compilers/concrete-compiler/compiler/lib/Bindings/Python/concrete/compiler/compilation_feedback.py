#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

"""Compilation feedback."""

import re
from typing import Dict, Set

# pylint: disable=no-name-in-module,import-error,too-many-instance-attributes
from mlir._mlir_libs._concretelang._compiler import (
    ProgramCompilationFeedback as _ProgramCompilationFeedback,
    CircuitCompilationFeedback as _CircuitCompilationFeedback,
    KeyType,
    PrimitiveOperation,
)

# pylint: enable=no-name-in-module,import-error

from .client_parameters import ClientParameters
from .parameter import Parameter
from .wrapper import WrapperCpp


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


class CircuitCompilationFeedback(WrapperCpp):
    """CircuitCompilationFeedback is a set of hint computed by the compiler engine for a circuit."""

    def __init__(self, circuit_compilation_feedback: _CircuitCompilationFeedback):
        """Wrap the native Cpp object.

        Args:
            circuit_compilation_feeback (_CircuitCompilationFeedback): object to wrap

        Raises:
            TypeError: if circuit_compilation_feedback is not of type _CircuitCompilationFeedback
        """
        if not isinstance(circuit_compilation_feedback, _CircuitCompilationFeedback):
            raise TypeError(
                "circuit_compilation_feedback must be of type "
                f"_CircuitCompilationFeedback, not {type(circuit_compilation_feedback)}"
            )

        self.name = circuit_compilation_feedback.name
        self.total_inputs_size = circuit_compilation_feedback.total_inputs_size
        self.total_output_size = circuit_compilation_feedback.total_output_size
        self.crt_decompositions_of_outputs = (
            circuit_compilation_feedback.crt_decompositions_of_outputs
        )
        self.statistics = circuit_compilation_feedback.statistics
        self.memory_usage_per_location = (
            circuit_compilation_feedback.memory_usage_per_location
        )

        super().__init__(circuit_compilation_feedback)

    def count(self, *, operations: Set[PrimitiveOperation]) -> int:
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
            for statistic in self.statistics
            if statistic.operation in operations
        )

    def count_per_parameter(
        self,
        *,
        operations: Set[PrimitiveOperation],
        key_types: Set[KeyType],
        client_parameters: ClientParameters,
    ) -> Dict[Parameter, int]:
        """
        Count the amount of specified operations in the program and group by parameters.

        Args:
            operations (Set[PrimitiveOperation]):
                set of operations used to filter the statistics

            key_types (Set[KeyType]):
                set of key types used to filter the statistics

            client_parameters (ClientParameters):
                client parameters required for grouping by parameters

        Returns:
            Dict[Parameter, int]:
                number of specified operations per parameter in the program
        """

        result = {}
        for statistic in self.statistics:
            if statistic.operation not in operations:
                continue

            for key_type, key_index in statistic.keys:
                if key_type not in key_types:
                    continue

                parameter = Parameter(client_parameters, key_type, key_index)
                if parameter not in result:
                    result[parameter] = 0
                result[parameter] += statistic.count

        return result

    def count_per_tag(self, *, operations: Set[PrimitiveOperation]) -> Dict[str, int]:
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
        for statistic in self.statistics:
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

    def count_per_tag_per_parameter(
        self,
        *,
        operations: Set[PrimitiveOperation],
        key_types: Set[KeyType],
        client_parameters: ClientParameters,
    ) -> Dict[str, Dict[Parameter, int]]:
        """
        Count the amount of specified operations in the program and group by tags and parameters.

        Args:
            operations (Set[PrimitiveOperation]):
                set of operations used to filter the statistics

            key_types (Set[KeyType]):
                set of key types used to filter the statistics

            client_parameters (ClientParameters):
                client parameters required for grouping by parameters

        Returns:
            Dict[str, Dict[Parameter, int]]:
                number of specified operations per tag per parameter in the program
        """

        result: Dict[str, Dict[int, int]] = {}
        for statistic in self.statistics:
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

                    parameter = Parameter(client_parameters, key_type, key_index)
                    if parameter not in result[current_tag]:
                        result[current_tag][parameter] = 0
                    result[current_tag][parameter] += statistic.count

        return result


class ProgramCompilationFeedback(WrapperCpp):
    """CompilationFeedback is a set of hint computed by the compiler engine."""

    def __init__(self, program_compilation_feedback: _ProgramCompilationFeedback):
        """Wrap the native Cpp object.

        Args:
            compilation_feeback (_CompilationFeedback): object to wrap

        Raises:
            TypeError: if program_compilation_feedback is not of type _CompilationFeedback
        """
        if not isinstance(program_compilation_feedback, _ProgramCompilationFeedback):
            raise TypeError(
                "program_compilation_feedback must be of type "
                f"_CompilationFeedback, not {type(program_compilation_feedback)}"
            )

        self.complexity = program_compilation_feedback.complexity
        self.p_error = program_compilation_feedback.p_error
        self.global_p_error = program_compilation_feedback.global_p_error
        self.total_secret_keys_size = (
            program_compilation_feedback.total_secret_keys_size
        )
        self.total_bootstrap_keys_size = (
            program_compilation_feedback.total_bootstrap_keys_size
        )
        self.total_keyswitch_keys_size = (
            program_compilation_feedback.total_keyswitch_keys_size
        )
        self.circuit_feedbacks = [
            CircuitCompilationFeedback(c)
            for c in program_compilation_feedback.circuit_feedbacks
        ]

        super().__init__(program_compilation_feedback)

    def circuit(self, circuit_name: str) -> CircuitCompilationFeedback:
        """
        Returns the feedback for the circuit circuit_name.

        Args:
            circuit_name (str):
                the name of the circuit.

        Returns:
            CircuitCompilationFeedback:
                the feedback for the circuit.

        Raises:
            TypeError: if the circuit_name is not a string
            ValueError: if there is no circuit with name circuit_name
        """
        if not isinstance(circuit_name, str):
            raise TypeError(
                f"circuit_name must be of type str, not {type(circuit_name)}"
            )

        for circuit in self.circuit_feedbacks:
            if circuit.name == circuit_name:
                return circuit

        raise ValueError(f"no circuit with name {circuit_name} found")

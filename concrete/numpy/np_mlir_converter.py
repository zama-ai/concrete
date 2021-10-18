"""Numpy-specific MLIR converter."""

from typing import Any, Dict

import numpy

from ..common.mlir.mlir_converter import MLIRConverter
from ..common.operator_graph import OPGraph
from ..common.representation.intermediate import UnivariateFunction


class NPMLIRConverter(MLIRConverter):
    """Numpy-specific MLIR converter."""

    @staticmethod
    def _generate_additional_info_dict(op_graph: OPGraph) -> Dict[str, Any]:
        """Generate the additional_conversion_info dict for the MLIR converter.

        Args:
            op_graph (OPGraph): the OPGraph for which we need the conversion infos.

        Returns:
            Dict[str, Any]: The dict with the additional conversion infos.
        """

        additional_conversion_info = {}

        # Disable numpy warnings during conversion to avoid issues during TLU generation
        with numpy.errstate(all="ignore"):
            additional_conversion_info["tables"] = {
                node: node.get_table()
                for node in op_graph.graph.nodes()
                if isinstance(node, UnivariateFunction)
            }

        return additional_conversion_info

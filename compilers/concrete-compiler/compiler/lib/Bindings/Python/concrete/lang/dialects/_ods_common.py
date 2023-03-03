#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

# We need this helpers from the mlir bindings, they are used in the generated files
from mlir.dialects._ods_common import (
    _cext,
    segmented_accessor,
    equally_sized_accessor,
    extend_opview_class,
    get_default_loc_context,
    get_op_result_or_value,
    get_op_results_or_values,
)

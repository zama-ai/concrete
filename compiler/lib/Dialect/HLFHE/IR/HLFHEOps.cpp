#include "mlir/IR/Region.h"

#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"


namespace mlir{
namespace zamalang{
    bool predApplyUnivariateRegion(::mlir::Region &region){
        if (region.getBlocks().size() != 1) {
            return false;
        }
        auto args = region.getBlocks().front().getArguments();
        if (args.size() != 1) {
            return false;
        }
        if (! args.front().getType().isa<mlir::IntegerType>()){
            return false;
        }
        //TODO: need to handle when there is no terminator
        auto terminator = region.getBlocks().front().getTerminator();
        return terminator->getName().getStringRef().equals("HLFHE.apply_univariate_return");
    }
}
}

#define GET_OP_CLASSES
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.cpp.inc"

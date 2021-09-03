#ifndef ZAMALANG_DIALECT_HLFHE_ANALYSIS_MANP
#define ZAMALANG_DIALECT_HLFHE_ANALYSIS_MANP

include "mlir/Pass/PassBase.td"

def MANP : FunctionPass<"MANP"> {
  let summary = "HLFHE Minimal Arithmetic Noise Padding Pass";
  let description = [{
   This pass calculates the Minimal Arithmetic Noise Padding
   (MANP) for each operation of a function and stores the result in an
   integer attribute named "MANP". This metric is identical to the
   ceiled 2-norm of the constant vector of an equivalent dot product
   between a vector of encrypted integers resulting directly from an
   encryption and a vector of plaintext constants.
  
   The pass supports the following operations:
  
    - HLFHE.dot_eint_int
    - HLFHE.zero
    - HLFHE.add_eint_int
    - HLFHE.add_eint
    - HLFHE.sub_int_eint
    - HLFHE.mul_eint_int
    - HLFHE.apply_lookup_table
  
   If any other operation is encountered, the pass conservatively
   fails. The pass further makes the optimistic assumption that all
   values passed to a function are either the direct result of an
   encryption of a noise-refreshing operation.
  
   Conceptually, the pass is equivalent to the three steps below:
  
     1. Replace all arithmetic operations with an equivalent dot
        operation
  
     2. Merge resulting dot operations into a single, equivalent
        dot operation
  
     3. Calculate the 2-norm of the vector of plaintext constants
        of the dot operation
  
   with the following replacement rules:
  
    - Function argument a      -> HLFHE.dot_eint_int([a], [1])
    - HLFHE.apply_lookup_table -> HLFHE.dot_eint_int([LUT result], [1])
    - HLFHE.zero()             -> HLFHE.dot_eint_int([encrypted 0], [1])
    - HLFHE.add_eint_int(e, c) -> HLFHE.dot_eint_int([e, 1], [1, c])
    - HLFHE.add_eint(e0, e1)   -> HLFHE.dot_eint_int([e0, e1], [1, 1])
    - HLFHE.sub_int_eint(c, e) -> HLFHE.dot_eint_int([e, c], [1, -1])
    - HLFHE.mul_eint_int(e, c) -> HLFHE.dot_eint_int([e], [c])
  
   Dependent dot operations, e.g.,
  
     a = HLFHE.dot_eint_int([a0, a1, ...], [c0, c1, ...])
     b = HLFHE.dot_eint_int([b0, b1, ...], [d0, d1, ...])
     x = HLFHE.dot_eint_int([a, b, ...], [f0, f1, ...])
  
   are merged as follows:
  
     x = HLFHE.dot_eint_int([a0, a1, ..., b0, b1, ...],
                            [f0*c0, f0*c1, ..., f1*d0, f1*d1, ...])
  
   However, the implementation does not explicitly create the
   equivalent dot operations, but only accumulates the squared 2-norm
   of the constant vector of the equivalent dot operation along the
   edges of the data-flow graph composed by the operations in order to
   calculate the final 2-norm for the final single dot operation above.
  
   For the example above, this means that the pass calculates the
   squared 2-norm of x, sqN(x) as:
  
     sqN(a) = c0*c0 + c1*c1 + ...
     sqN(b) = d0*d0 + d1*d1 + ...
     sqN(x) = f0*f0*c0*c0 + f0*f0*c1*c1 + ... + f1*f1*d0*d0 + f1*f1*d1*d1 + ...
            = f0*f0*sqN(a) + f1*f1*sqN(b)
  
   This leads to the following rules to calculate the squared 2-norm
   for the supported operations:
  
    - Function argument        -> 1
    - HLFHE.apply_lookup_table -> 1
    - HLFHE.zero()             -> 1
    - HLFHE.dot_eint_int([e0, e1, ...], [c0, c1, ...]) ->
         c0*c0*sqN(e0) + c1*c1*sqN(e1) + ...
    - HLFHE.add_eint_int(e, c) -> 1*1*sqN(e) + c*c*1*1 = sqN(e) + c*c
    - HLFHE.add_eint(e0, e1)   -> 1*1*sqN(e0) + 1*1*sqN(e2) = sqN(e1) + sqN(e2)
    - HLFHE.sub_int_eint(c, e) -> 1*1*sqN(e) + c*c*(-1)*(-1) = sqN(e) + c*c
    - HLFHE.mul_eint_int(e, c) -> c*c*sqN(e)

   The final, non-squared 2-norm of an operation is the square root of the
   squared value rounded to the next highest integer.
  }];
}

#endif

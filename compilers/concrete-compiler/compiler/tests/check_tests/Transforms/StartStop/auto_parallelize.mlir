// This test check that the start and stop dfr operations are correctly setup according to the options
// RUN: concretecompiler --action=dump-std --parallelize --passes=dfr-add-start-stop --skip-program-info %s 2>&1| FileCheck %s

func.func @bar() -> () {
    // CHECK: call @_dfr_start(%c1_i64, %true, %c0_i64) : (i64, i1, i64) -> ()
    // CHECK: call @_dfr_stop(%c1_i64) : (i64) -> ()
    return
}

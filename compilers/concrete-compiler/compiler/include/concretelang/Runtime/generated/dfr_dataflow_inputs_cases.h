case 0:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx]() -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    }));
break;

case 1:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future));
break;

case 2:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future));
break;

case 3:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future));
break;

case 4:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get(),
                                    param3.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future));
break;

case 5:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get(),
                                    param3.get(), param4.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future));
break;

case 6:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get(),
                                    param3.get(), param4.get(), param5.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future));
break;

case 7:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
          hpx::shared_future<void *> param6)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get(),
                                    param3.get(), param4.get(), param5.get(),
                                    param6.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future));
break;

case 8:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
          hpx::shared_future<void *> param6, hpx::shared_future<void *> param7)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get(),
                                    param3.get(), param4.get(), param5.get(),
                                    param6.get(), param7.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future));
break;

case 9:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
          hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
          hpx::shared_future<void *> param8)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get(),
                                    param3.get(), param4.get(), param5.get(),
                                    param6.get(), param7.get(), param8.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future));
break;

case 10:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
          hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
          hpx::shared_future<void *> param8, hpx::shared_future<void *> param9)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(), param1.get(), param2.get(), param3.get(), param4.get(),
          param5.get(), param6.get(), param7.get(), param8.get(), param9.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future));
break;

case 11:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
          hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
          hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
          hpx::shared_future<void *> param10)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(), param2.get(),
                                    param3.get(), param4.get(), param5.get(),
                                    param6.get(), param7.get(), param8.get(),
                                    param9.get(), param10.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future));
break;

case 12:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
          hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
          hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
          hpx::shared_future<void *> param10,
          hpx::shared_future<void *> param11)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(),  param2.get(),
                                    param3.get(), param4.get(),  param5.get(),
                                    param6.get(), param7.get(),  param8.get(),
                                    param9.get(), param10.get(), param11.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future));
break;

case 13:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
          hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
          hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
          hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
          hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
          hpx::shared_future<void *> param10,
          hpx::shared_future<void *> param11,
          hpx::shared_future<void *> param12)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(), param1.get(),  param2.get(),
                                    param3.get(), param4.get(),  param5.get(),
                                    param6.get(), param7.get(),  param8.get(),
                                    param9.get(), param10.get(), param11.get(),
                                    param12.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future));
break;

case 14:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(),  param1.get(),  param2.get(),
                                    param3.get(),  param4.get(),  param5.get(),
                                    param6.get(),  param7.get(),  param8.get(),
                                    param9.get(),  param10.get(), param11.get(),
                                    param12.get(), param13.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future));
break;

case 15:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future));
break;

case 16:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future));
break;

case 17:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {param0.get(),  param1.get(),  param2.get(),
                                    param3.get(),  param4.get(),  param5.get(),
                                    param6.get(),  param7.get(),  param8.get(),
                                    param9.get(),  param10.get(), param11.get(),
                                    param12.get(), param13.get(), param14.get(),
                                    param15.get(), param16.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future));
break;

case 18:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future));
break;

case 19:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future));
break;

case 20:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future));
break;

case 21:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future));
break;

case 22:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future));
break;

case 23:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future));
break;

case 24:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future));
break;

case 25:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future));
break;

case 26:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future));
break;

case 27:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future));
break;

case 28:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future));
break;

case 29:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future));
break;

case 30:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future));
break;

case 31:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future));
break;

case 32:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future));
break;

case 33:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future));
break;

case 34:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future));
break;

case 35:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future));
break;

case 36:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future));
break;

case 37:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future));
break;

case 38:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future));
break;

case 39:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future));
break;

case 40:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future));
break;

case 41:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future));
break;

case 42:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future));
break;

case 43:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future));
break;

case 44:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42, hpx::shared_future<void *> param43)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get(), param43.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future,
    *((dfr_refcounted_future_p)refcounted_futures[43])->future));
break;

case 45:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42, hpx::shared_future<void *> param43,
        hpx::shared_future<void *> param44)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get(), param43.get(),
          param44.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future,
    *((dfr_refcounted_future_p)refcounted_futures[43])->future,
    *((dfr_refcounted_future_p)refcounted_futures[44])->future));
break;

case 46:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42, hpx::shared_future<void *> param43,
        hpx::shared_future<void *> param44, hpx::shared_future<void *> param45)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get(), param43.get(),
          param44.get(), param45.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future,
    *((dfr_refcounted_future_p)refcounted_futures[43])->future,
    *((dfr_refcounted_future_p)refcounted_futures[44])->future,
    *((dfr_refcounted_future_p)refcounted_futures[45])->future));
break;

case 47:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42, hpx::shared_future<void *> param43,
        hpx::shared_future<void *> param44, hpx::shared_future<void *> param45,
        hpx::shared_future<void *> param46)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get(), param43.get(),
          param44.get(), param45.get(), param46.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future,
    *((dfr_refcounted_future_p)refcounted_futures[43])->future,
    *((dfr_refcounted_future_p)refcounted_futures[44])->future,
    *((dfr_refcounted_future_p)refcounted_futures[45])->future,
    *((dfr_refcounted_future_p)refcounted_futures[46])->future));
break;

case 48:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42, hpx::shared_future<void *> param43,
        hpx::shared_future<void *> param44, hpx::shared_future<void *> param45,
        hpx::shared_future<void *> param46, hpx::shared_future<void *> param47)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get(), param43.get(),
          param44.get(), param45.get(), param46.get(), param47.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future,
    *((dfr_refcounted_future_p)refcounted_futures[43])->future,
    *((dfr_refcounted_future_p)refcounted_futures[44])->future,
    *((dfr_refcounted_future_p)refcounted_futures[45])->future,
    *((dfr_refcounted_future_p)refcounted_futures[46])->future,
    *((dfr_refcounted_future_p)refcounted_futures[47])->future));
break;

case 49:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42, hpx::shared_future<void *> param43,
        hpx::shared_future<void *> param44, hpx::shared_future<void *> param45,
        hpx::shared_future<void *> param46, hpx::shared_future<void *> param47,
        hpx::shared_future<void *> param48)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get(), param43.get(),
          param44.get(), param45.get(), param46.get(), param47.get(),
          param48.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future,
    *((dfr_refcounted_future_p)refcounted_futures[43])->future,
    *((dfr_refcounted_future_p)refcounted_futures[44])->future,
    *((dfr_refcounted_future_p)refcounted_futures[45])->future,
    *((dfr_refcounted_future_p)refcounted_futures[46])->future,
    *((dfr_refcounted_future_p)refcounted_futures[47])->future,
    *((dfr_refcounted_future_p)refcounted_futures[48])->future));
break;

case 50:
oodf = std::move(hpx::dataflow(
    [wfnname, param_sizes, param_types, output_sizes, output_types, gcc_target,
     ctx](
        hpx::shared_future<void *> param0, hpx::shared_future<void *> param1,
        hpx::shared_future<void *> param2, hpx::shared_future<void *> param3,
        hpx::shared_future<void *> param4, hpx::shared_future<void *> param5,
        hpx::shared_future<void *> param6, hpx::shared_future<void *> param7,
        hpx::shared_future<void *> param8, hpx::shared_future<void *> param9,
        hpx::shared_future<void *> param10, hpx::shared_future<void *> param11,
        hpx::shared_future<void *> param12, hpx::shared_future<void *> param13,
        hpx::shared_future<void *> param14, hpx::shared_future<void *> param15,
        hpx::shared_future<void *> param16, hpx::shared_future<void *> param17,
        hpx::shared_future<void *> param18, hpx::shared_future<void *> param19,
        hpx::shared_future<void *> param20, hpx::shared_future<void *> param21,
        hpx::shared_future<void *> param22, hpx::shared_future<void *> param23,
        hpx::shared_future<void *> param24, hpx::shared_future<void *> param25,
        hpx::shared_future<void *> param26, hpx::shared_future<void *> param27,
        hpx::shared_future<void *> param28, hpx::shared_future<void *> param29,
        hpx::shared_future<void *> param30, hpx::shared_future<void *> param31,
        hpx::shared_future<void *> param32, hpx::shared_future<void *> param33,
        hpx::shared_future<void *> param34, hpx::shared_future<void *> param35,
        hpx::shared_future<void *> param36, hpx::shared_future<void *> param37,
        hpx::shared_future<void *> param38, hpx::shared_future<void *> param39,
        hpx::shared_future<void *> param40, hpx::shared_future<void *> param41,
        hpx::shared_future<void *> param42, hpx::shared_future<void *> param43,
        hpx::shared_future<void *> param44, hpx::shared_future<void *> param45,
        hpx::shared_future<void *> param46, hpx::shared_future<void *> param47,
        hpx::shared_future<void *> param48, hpx::shared_future<void *> param49)
        -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
      std::vector<void *> params = {
          param0.get(),  param1.get(),  param2.get(),  param3.get(),
          param4.get(),  param5.get(),  param6.get(),  param7.get(),
          param8.get(),  param9.get(),  param10.get(), param11.get(),
          param12.get(), param13.get(), param14.get(), param15.get(),
          param16.get(), param17.get(), param18.get(), param19.get(),
          param20.get(), param21.get(), param22.get(), param23.get(),
          param24.get(), param25.get(), param26.get(), param27.get(),
          param28.get(), param29.get(), param30.get(), param31.get(),
          param32.get(), param33.get(), param34.get(), param35.get(),
          param36.get(), param37.get(), param38.get(), param39.get(),
          param40.get(), param41.get(), param42.get(), param43.get(),
          param44.get(), param45.get(), param46.get(), param47.get(),
          param48.get(), param49.get()};
      mlir::concretelang::dfr::OpaqueInputData oid(wfnname, params, param_sizes,
                                                   param_types, output_sizes,
                                                   output_types, ctx);
      return gcc_target->execute_task(oid);
    },
    *((dfr_refcounted_future_p)refcounted_futures[0])->future,
    *((dfr_refcounted_future_p)refcounted_futures[1])->future,
    *((dfr_refcounted_future_p)refcounted_futures[2])->future,
    *((dfr_refcounted_future_p)refcounted_futures[3])->future,
    *((dfr_refcounted_future_p)refcounted_futures[4])->future,
    *((dfr_refcounted_future_p)refcounted_futures[5])->future,
    *((dfr_refcounted_future_p)refcounted_futures[6])->future,
    *((dfr_refcounted_future_p)refcounted_futures[7])->future,
    *((dfr_refcounted_future_p)refcounted_futures[8])->future,
    *((dfr_refcounted_future_p)refcounted_futures[9])->future,
    *((dfr_refcounted_future_p)refcounted_futures[10])->future,
    *((dfr_refcounted_future_p)refcounted_futures[11])->future,
    *((dfr_refcounted_future_p)refcounted_futures[12])->future,
    *((dfr_refcounted_future_p)refcounted_futures[13])->future,
    *((dfr_refcounted_future_p)refcounted_futures[14])->future,
    *((dfr_refcounted_future_p)refcounted_futures[15])->future,
    *((dfr_refcounted_future_p)refcounted_futures[16])->future,
    *((dfr_refcounted_future_p)refcounted_futures[17])->future,
    *((dfr_refcounted_future_p)refcounted_futures[18])->future,
    *((dfr_refcounted_future_p)refcounted_futures[19])->future,
    *((dfr_refcounted_future_p)refcounted_futures[20])->future,
    *((dfr_refcounted_future_p)refcounted_futures[21])->future,
    *((dfr_refcounted_future_p)refcounted_futures[22])->future,
    *((dfr_refcounted_future_p)refcounted_futures[23])->future,
    *((dfr_refcounted_future_p)refcounted_futures[24])->future,
    *((dfr_refcounted_future_p)refcounted_futures[25])->future,
    *((dfr_refcounted_future_p)refcounted_futures[26])->future,
    *((dfr_refcounted_future_p)refcounted_futures[27])->future,
    *((dfr_refcounted_future_p)refcounted_futures[28])->future,
    *((dfr_refcounted_future_p)refcounted_futures[29])->future,
    *((dfr_refcounted_future_p)refcounted_futures[30])->future,
    *((dfr_refcounted_future_p)refcounted_futures[31])->future,
    *((dfr_refcounted_future_p)refcounted_futures[32])->future,
    *((dfr_refcounted_future_p)refcounted_futures[33])->future,
    *((dfr_refcounted_future_p)refcounted_futures[34])->future,
    *((dfr_refcounted_future_p)refcounted_futures[35])->future,
    *((dfr_refcounted_future_p)refcounted_futures[36])->future,
    *((dfr_refcounted_future_p)refcounted_futures[37])->future,
    *((dfr_refcounted_future_p)refcounted_futures[38])->future,
    *((dfr_refcounted_future_p)refcounted_futures[39])->future,
    *((dfr_refcounted_future_p)refcounted_futures[40])->future,
    *((dfr_refcounted_future_p)refcounted_futures[41])->future,
    *((dfr_refcounted_future_p)refcounted_futures[42])->future,
    *((dfr_refcounted_future_p)refcounted_futures[43])->future,
    *((dfr_refcounted_future_p)refcounted_futures[44])->future,
    *((dfr_refcounted_future_p)refcounted_futures[45])->future,
    *((dfr_refcounted_future_p)refcounted_futures[46])->future,
    *((dfr_refcounted_future_p)refcounted_futures[47])->future,
    *((dfr_refcounted_future_p)refcounted_futures[48])->future,
    *((dfr_refcounted_future_p)refcounted_futures[49])->future));
break;

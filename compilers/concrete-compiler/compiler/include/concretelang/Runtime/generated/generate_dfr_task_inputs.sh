#!/bin/bash

# Param ranges $1 - $2 for the number of inputs (e.g., 0 50)

p1=""
p2=""
p3=""
for i in $(eval echo {$1..$2}); do
    if ((i == 1)); then
	p1="$p1 hpx::shared_future<void *> param$(($i-1))"
	p2="$p2 param$(($i-1)).get()"
	p3="$p3, *((dfr_refcounted_future_p)refcounted_futures[$(($i-1))])->future"
    fi
    if ((i > 1)); then
	p1="$p1, hpx::shared_future<void *> param$(($i-1))"
	p2="$p2, param$(($i-1)).get()"
	p3="$p3, *((dfr_refcounted_future_p)refcounted_futures[$(($i-1))])->future"
    fi
    echo "case $i:
    	 oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx]($p1)"
    echo "-> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {$p2};"
    echo "          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        } $p3));
    	 break;
	 "
done;

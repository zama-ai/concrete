#!/bin/bash

# Param ranges $1 - $2 for the number of outputs (e.g., 1 3)
# $3 - $4 for inputs (e.g., 0 50)

for i in $(eval echo {$1..$2}); do
    outs=""
    echo "case $i: {"
    for j in $(eval echo {1..$i}); do
	echo " void *output$j;
	      _dfr_checked_aligned_alloc(&output$j, 512, inputs.output_sizes[$(($j-1))]);"
	if ((j == 1)); then
	   outs="$outs output$j"
	else
	    outs="$outs, output$j"
	fi
    done;
    echo "      switch (inputs.params.size()) {"

    ins=""
    for j in $(eval echo {$3..$4}); do
	if ((j > 0)); then
		ins="$ins, inputs.params[$(($j - 1))]"
	fi
	echo "case $j:
	     wfn($outs$ins); break;"
    done
    echo "      default:
        HPX_THROW_EXCEPTION(hpx::error::no_success,
                            \"GenericComputeServer::execute_task\",
                            \"Error: number of task parameters not supported.\");
      }"
    echo "outputs = {$outs}; break;}"
done;

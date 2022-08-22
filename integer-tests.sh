#!/bin/bash

set -e

# block pbs are too slow for high params
# mul_crt_4_4 is extremely flaky (~80% failure)
filter_expression=''\
'not test(/.*_block_pbs(_base)?_param_message_[34]_carry_[34]$/)'\
'and not test(~mul_crt_param_message_4_carry_4)'

export RUSTFLAGS="-C target-cpu=native"

cargo nextest run \
    --release \
    --package concrete-integer \
    --profile ci \
    --features internal-keycache \
    --test-threads 14 \
    -E "$filter_expression"

cargo test \
    --release \
    --package concrete-integer \
    --features internal-keycache \
    --doc

echo "Test ran in $SECONDS seconds"

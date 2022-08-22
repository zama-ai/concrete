#!/bin/bash

set -e

filter_expression=''\
'('\
'   test(/^server_key::.*_param_message_1_carry_1$/)'\
'or test(/^server_key::.*_param_message_1_carry_2$/)'\
'or test(/^server_key::.*_param_message_1_carry_3$/)'\
'or test(/^server_key::.*_param_message_1_carry_4$/)'\
'or test(/^server_key::.*_param_message_1_carry_5$/)'\
'or test(/^server_key::.*_param_message_1_carry_6$/)'\
'or test(/^server_key::.*_param_message_2_carry_2$/)'\
'or test(/^server_key::.*_param_message_2_carry_2$/)'\
'or test(/^server_key::.*_param_message_3_carry_3$/)'\
'or test(/^server_key::.*_param_message_4_carry_4$/)'\
'or test(/^treepbs::.*_param_message_1_carry_1$/)'\
'or test(/^treepbs::.*_param_message_2_carry_2$/)'\
')'\
'and not test(~smart_add_and_mul)' # This test is too slow

export RUSTFLAGS="-C target-cpu=native"

cargo nextest run \
    --release \
    --package concrete-shortint \
    --profile ci \
    --features internal-keycache \
    --test-threads 10 \
    -E "$filter_expression"

cargo test \
    --release \
    --package concrete-shortint \
    --features internal-keycache \
    --doc

echo "Test ran in $SECONDS seconds"

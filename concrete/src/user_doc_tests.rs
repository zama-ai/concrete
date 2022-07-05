use doc_comment::doctest;

doctest!("../docs/getting_started/quick_start.md", quick_start);
doctest!("../docs/getting_started/limitations.md", limitations);
doctest!(
    "../docs/getting_started/operations_and_examples.md",
    operations_and_examples
);
doctest!("../docs/how_to/dynamic_types.md", dynamic_types);
doctest!("../docs/how_to/generic_bounds.md", generic_bounds);
doctest!("../docs/how_to/serialization.md", serialization);
doctest!("../docs/tutorials/parity_bit.md", parity_bit);
doctest!("../docs/tutorials/latin_string.md", latin_string);
doctest!("../README.md", readme);

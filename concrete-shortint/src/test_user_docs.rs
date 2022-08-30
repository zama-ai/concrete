use doc_comment::doctest;

doctest!("../docs/getting_started/quick_start.md", quick_start);
doctest!("../docs/getting_started/operations.md", operations);
doctest!("../docs/how_to/pbs.md", pbs);
doctest!("../docs/tutorials/serialization.md", serialization_tuto);
doctest!(
    "../docs/tutorials/circuit_evaluation.md",
    circuit_evaluation
);

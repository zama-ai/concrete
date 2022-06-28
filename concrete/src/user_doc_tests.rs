use doc_comment::doctest;

// We add all the files even if does not currently
// have any doctest

doctest!("../docs/user/things_to_know.md");
doctest!("../docs/user/how_to.md");
doctest!("../docs/user/tutorial.md");
doctest!("../docs/user/getting_started.md");

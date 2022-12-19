CURVES_JSON_PATH=json/curves.json
CURVES_CPP_GEN_H=concrete-security-curves-cpp/include/concrete/curves.gen.h
CURVES_RUST_GEN_TXT=concrete-security-curves-rust/src/curves.gen.rs

$(CURVES_JSON_PATH): verify_curves.py
	sage verify_curves.py > $@

$(CURVES_CPP_GEN_H): concrete-security-curves-cpp/gen_header.py $(CURVES_JSON_PATH)
	cat $(CURVES_JSON_PATH) | python concrete-security-curves-cpp/gen_header.py > $(CURVES_CPP_GEN_H)

generate-cpp: $(CURVES_CPP_GEN_H)

$(CURVES_RUST_GEN_TXT): rust/gen_table.py $(CURVES_JSON_PATH)
	cat $(CURVES_JSON_PATH) | python concrete-security-curves-cpp/gen_table.py > $(CURVES_CPP_GEN_H)

generate-rust: $(CURVES_RUST_GEN_TXT)

.PHONY: generate-cpp-header
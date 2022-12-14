CURVES_JSON_PATH=json/curves.json
CURVES_CPP_GEN_H=cpp/include/concrete/curves.gen.h

$(CURVES_JSON_PATH): verify_curves.py
	sage verify_curves.py > $@

$(CURVES_CPP_GEN_H): cpp/gen_header.py $(CURVES_JSON_PATH)
	cat $(CURVES_JSON_PATH) | python cpp/gen_header.py > $(CURVES_CPP_GEN_H)

generate-cpp-header: $(CURVES_CPP_GEN_H)

.PHONY: generate-cpp-header
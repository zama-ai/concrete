COMPILER_BUILD_DIR?=$(realpath ../compiler)/build/

update_doc_version:
	sed -i "s/release = '.*'/release = '`git describe --tags --abbrev=0`'/g" conf.py

# This depends on calling make doc in the compiler project first
doc: update_doc_version update_doxygen_conf
	bash symlink_md.bash ${COMPILER_BUILD_DIR}
	bash symlink_py.bash ${COMPILER_BUILD_DIR}
	sphinx-apidoc -o python_api/ links_to_compiler_build/py/concretelang_core/concrete -f
	PYTHONPATH=links_to_compiler_build/py/concretelang_core sphinx-build -M html . _build

update_doxygen_conf:
	sed "s#../compiler/build#${COMPILER_BUILD_DIR}#g" -i Doxyfile.in

clean:
	rm -rf _build/
	rm -rf links_to_compiler_build/
	rm -rf python_api/*

.PHONY: update_doc_version

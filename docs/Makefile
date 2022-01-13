update_doc_version:
	sed -i "s/release = '.*'/release = '`git describe --tags --abbrev=0`'/g" conf.py

doc: update_doc_version
	cd ../compiler; make doc python-bindings
	bash symlink_md.bash
	bash symlink_py.bash
	sphinx-apidoc -o python_api/ links_to_compiler_build/py/concretelang_core/concrete -f
	PYTHONPATH=links_to_compiler_build/py/concretelang_core sphinx-build -M html . _build

clean:
	rm -rf _build/
	rm -rf links_to_compiler_build/
	rm -rf python_api/*

.PHONY: update_doc_version
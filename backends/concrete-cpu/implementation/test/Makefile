ROOT_DIR=$(realpath $(PWD)/..)

regenererate_lib:
	# rm -rf ../include/concrete-cpu.h
	cargo build

test_encryption: regenererate_lib
	zig test test_encryption.zig -I$(ROOT_DIR) -lc -lconcrete_cpu -lunwind -L../target/debug/


test_bootstrap: regenererate_lib
	zig test test_bootstrap.zig -I$(ROOT_DIR) -lc -lconcrete_cpu -lunwind -L../target/debug/

test_vertical_packing: regenererate_lib
	zig test test_vertical_packing.zig -I$(ROOT_DIR) -lc -lconcrete_cpu -lunwind -L../target/debug/


test_bit_extract: regenererate_lib
	zig test test_bit_extract.zig -I$(ROOT_DIR) -lc -lconcrete_cpu -lunwind -L../target/debug/

test: test_encryption test_bootstrap test_bit_extract test_vertical_packing 

test:
	# rm -rf ../include/concrete-cpu.h
	cargo build
	zig test all.zig -I../include -lunwind -lconcrete_cpu -L../target/debug -framework Security

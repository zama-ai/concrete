CC	 := gcc # gcc only has openmp support on test machine
CARGS = -std=c99 ../src/hcs_random.c ../src/com/util.c ../src/com/parson.c \
		../src/com/ripemd160.c -lgmp -Ofast -march=native

all:

s_pcs_encrypt:
	$(CC) $(CARGS) pcs_encrypt.c ../src/pcs.c

p_pcs_encrypt:
	$(CC) $(CARGS) pcs_encrypt.c ../src/pcs.c -fopenmp

s_pcs_decrypt:
	$(CC) $(CARGS) pcs_decrypt.c ../src/pcs.c

p_pcs_decrypt:
	$(CC) $(CARGS) pcs_decrypt.c ../src/pcs.c -fopenmp

s_egcs_ee_mul:
	$(CC) $(CARGS) pcs_decrypt.c ../src/egcs.c

p_egcs_ee_mul:
	$(CC) $(CARGS) pcs_decrypt.c ../src/egcs.c -fopenmp

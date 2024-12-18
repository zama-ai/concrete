# TFHE-rs interoperability example

This is the full execution for the example explained in the [TFHE-rs Interoperability Guide](../../../../docs/guides/tfhers) (use case 1). You can find the TFHE-rs code [here](../../tests/tfhers-utils/src/main.rs), while the Python code is under this direcotry [here](example.py). Both are CLI tools, so that we can execute the example step by step. You can refer to the code at every step to see how it's implemented.

## Make tmpdir

We want to setup a temporary working directory first:

```sh
export TDIR=`mktemp -d`
```

## KeyGen

First we need to build the TFHE-rs utility in [this directory](../../tests/tfhers-utils/) by running the following:

```sh
cd ../../tests/tfhers-utils/
make build
cd -
```

Then we can generate keys in two different ways. You only need to run one of the following methods.

#### Generate the Secret Key in Concrete

We start by doing keygen in Concrete:

```sh
python example.py keygen -o $TDIR/concrete_sk -k $TDIR/concrete_keyset
```

Then we do a partial keygen in TFHE-rs:

```sh
../../tests/tfhers-utils/target/release/tfhers_utils keygen --lwe-sk $TDIR/concrete_sk --output-lwe-sk $TDIR/tfhers_sk -c $TDIR/tfhers_client_key -s $TDIR/tfhers_server_key
```

#### Generate the Secret Key in TFHE-rs

We start by doing keygen in TFHE-rs:

```sh
../../tests/tfhers-utils/target/release/tfhers_utils keygen --output-lwe-sk $TDIR/tfhers_sk -c $TDIR/tfhers_client_key -s $TDIR/tfhers_server_key
```

Then we do a partial keygen in Concrete:

```sh
python example.py keygen -s $TDIR/tfhers_sk -o $TDIR/concrete_sk -k $TDIR/concrete_keyset
```

## Encrypt in TFHE-rs

```sh
../../tests/tfhers-utils/target/release/tfhers_utils encrypt-with-key --value 162 --ciphertext $TDIR/tfhers_ct_1 --client-key $TDIR/tfhers_client_key
../../tests/tfhers-utils/target/release/tfhers_utils encrypt-with-key --value 73 --ciphertext $TDIR/tfhers_ct_2 --client-key $TDIR/tfhers_client_key
```

{% hint style="info" %}

If you have tensor inputs, then you can encrypt by passing your flat tensor in `--value`. Concrete will take care of reshaping the values to the corresponding shape. For example `--value=1,2,3,4` can represent a 2 by 2 tensor, or a flat vector of 4 values.

{% endhint %}

## Compute in TFHE-rs

```sh
# encrypt value to add first
../../tests/tfhers-utils/target/release/tfhers_utils encrypt-with-key --value 9 --ciphertext $TDIR/tfhers_ct_inc --client-key $TDIR/tfhers_client_key
# add two ciphertexts
../../tests/tfhers-utils/target/release/tfhers_utils add --server-key $TDIR/tfhers_server_key --cts $TDIR/tfhers_ct_2 $TDIR/tfhers_ct_inc --output-ct $TDIR/tfhers_ct_2
```

## Run in Concrete

```sh
python example.py run -k $TDIR/concrete_keyset -c1 $TDIR/tfhers_ct_1 -c2 $TDIR/tfhers_ct_2 -o $TDIR/tfhers_ct_out
```

## Decrypt in TFHE-rs

```sh
../../tests/tfhers-utils/target/release/tfhers_utils decrypt-with-key --ciphertext $TDIR/tfhers_ct_out --client-key $TDIR/tfhers_client_key
```

## Clean tmpdir

```sh
rm -rf $TDIR
```

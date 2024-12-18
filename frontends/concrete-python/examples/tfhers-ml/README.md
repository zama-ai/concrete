# TFHE-rs interoperability example

This is a similar example to the [first TFHE-rs example](../tfhers/), except that it uses tensors and run a linear ML model. It also uses quantization.

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

## Quantize values

We need to quantize floating point inputs using a pre-built quantizer for our ML model:

```sh
../../tests/tfhers-utils/target/release/tfhers_utils quantize --value=5.1,3.5,1.4,0.2,4.9,3,1.4,0.2,4.7,3.2,1.3,0.2,4.6,3.1,1.5,0.2,5,3.6,1.4,0.2 --config ./input_quantizer.json -o $TDIR/quantized_values
```

## Encrypt in TFHE-rs

```sh
../../tests/tfhers-utils/target/release/tfhers_utils encrypt-with-key --signed --value=$(cat $TDIR/quantized_values) --ciphertext $TDIR/tfhers_ct --client-key $TDIR/tfhers_client_key
```

## Run in Concrete

```sh
python example.py run -k $TDIR/concrete_keyset -c $TDIR/tfhers_ct -o $TDIR/tfhers_ct_out
```

## Decrypt in TFHE-rs

```sh
../../tests/tfhers-utils/target/release/tfhers_utils decrypt-with-key --tensor --signed --ciphertext $TDIR/tfhers_ct_out --client-key $TDIR/tfhers_client_key --plaintext $TDIR/result_plaintext
```

## Rescale Output

At the end of the circuit, we are rounding the result to 8 bits, discarding the remaining LSB bits. As we have `lsbs_to_remove=10` we are re-introducing the 10 bits of LSB.

```sh
python -c "print(','.join(map(lambda x: str(x << 10), [$(cat $TDIR/result_plaintext)])))" > $TDIR/rescaled_plaintext
```


## Dequantize values

We need to dequantize integer outputs using a pre-built quantizer for our ML model:

```sh
../../tests/tfhers-utils/target/release/tfhers_utils dequantize --value=$(cat $TDIR/rescaled_plaintext) --shape=5,3 --config ./output_quantizer.json
```

## Compute error

We compare the output to the expected result:

```sh
python compute_error.py --plaintext-file "$TDIR/rescaled_plaintext" --quantized-predictions-file "test_values.json"
```

## Clean tmpdir

```sh
rm -rf $TDIR
```

import hashlib
import time
from copy import deepcopy

import numpy as np

from concrete import fhe

CHUNK_WIDTH = 2

assert CHUNK_WIDTH in {1, 2, 4, 8}

NUMBER_OF_CHUNKS = 32 // CHUNK_WIDTH


def number_to_chunks(number, size=32):
    bit_pattern = np.binary_repr(number, width=size)

    result = []
    for i in range(0, len(bit_pattern), CHUNK_WIDTH):
        chunk = int(bit_pattern[i : (i + CHUNK_WIDTH)], 2)
        result.append(chunk)
    return np.array(result)


def chunks_to_number(chunks):
    result = 0
    for i, chunk in enumerate(reversed(chunks)):
        result += int(chunk) << (CHUNK_WIDTH * i)
    return result


K = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]
H = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]


K_CHUNKED = [number_to_chunks(K_i) for K_i in K]
H_CHUNKED = [number_to_chunks(H_i) for H_i in H]


def add(*chunked_numbers):
    carry = 0
    raw_addition = sum(chunked_numbers)

    results = []
    for i in reversed(range(NUMBER_OF_CHUNKS)):
        result_with_overflow = raw_addition[i] + carry

        carry = result_with_overflow >> CHUNK_WIDTH
        result = result_with_overflow - (carry * (2**CHUNK_WIDTH))

        results.append(result)
    results = list(reversed(results))

    if all(isinstance(x, (int, np.integer)) for x in results):
        return np.array(results)

    return fhe.array(results)


def right_rotate_chunks(chunks, amount):
    return np.concatenate((chunks[-amount:], chunks[:-amount]))


def right_shift_chunks(chunks, amount):
    return np.concatenate(([0] * chunks[-amount:].shape[0], chunks[:-amount]))


def right_rotate_number(chunks, amount):
    n = amount // CHUNK_WIDTH
    m = amount % CHUNK_WIDTH

    if n != 0:
        rotated_chunks = right_rotate_chunks(chunks, amount=n)
    else:
        rotated_chunks = chunks

    if m != 0:
        raise_low_bits = lambda x: (x % 2**m) << (CHUNK_WIDTH - m)
        raised_low_bits = fhe.univariate(raise_low_bits)(rotated_chunks)
        shifted_raised_low_bits = right_rotate_chunks(raised_low_bits, amount=1)
        high_bits = rotated_chunks >> m
        rotated_number = shifted_raised_low_bits + high_bits
    else:
        rotated_number = rotated_chunks

    return rotated_number


def right_shift_number(chunks, amount):
    n = amount // CHUNK_WIDTH
    m = amount % CHUNK_WIDTH

    if n != 0:
        shifted_chunks = right_shift_chunks(chunks, n)
    else:
        shifted_chunks = chunks

    if m != 0:
        raise_low_bits = lambda x: (x % 2**m) << (CHUNK_WIDTH - m)
        raised_low_bits = fhe.univariate(raise_low_bits)(shifted_chunks)
        shifted_raised_low_bits = right_shift_chunks(raised_low_bits, 1)
        high_bits = shifted_chunks >> m
        result = shifted_raised_low_bits + high_bits
    else:
        result = shifted_chunks

    return result


def s0(w):
    return right_rotate_number(w, 7) ^ right_rotate_number(w, 18) ^ right_shift_number(w, 3)


def s1(w):
    return right_rotate_number(w, 17) ^ right_rotate_number(w, 19) ^ right_shift_number(w, 10)


def S0(a_word):
    return (
        right_rotate_number(a_word, 2)
        ^ right_rotate_number(a_word, 13)
        ^ right_rotate_number(a_word, 22)
    )


def S1(e_word):
    return (
        right_rotate_number(e_word, 6)
        ^ right_rotate_number(e_word, 11)
        ^ right_rotate_number(e_word, 25)
    )


def Ch(e_word, f_word, g_word):
    return (e_word & f_word) ^ ((2**CHUNK_WIDTH - 1 - e_word) & g_word)


def Maj(a_word, b_word, c_word):
    return (a_word & b_word) ^ (a_word & c_word) ^ (b_word & c_word)


def state_update(state, w_i_plus_k_i):
    a, b, c, d, e, f, g, h = state

    temp1 = add(h, S1(e), Ch(e, f, g), w_i_plus_k_i)
    temp2 = add(S0(a), Maj(a, b, c))

    new_a = add(temp1, temp2)
    new_e = add(d, temp1)

    return [new_a, a, b, c, new_e, e, f, g]


def sha256(data, number_of_rounds=64):
    h_chunks = deepcopy(H_CHUNKED)
    k_chunks = deepcopy(K_CHUNKED)

    for i in range(0, (data.shape[0] * 32) // 512):
        with fhe.tag(f"input[{i * 16 * 4}:{(i + 1) * 16 * 4}]"):
            with fhe.tag("extraction"):
                start = i * 16
                end = start + 16
                chunks = data[start:end]

                state = h_chunks

            w = [None for _ in range(number_of_rounds)]
            for j in range(0, number_of_rounds):
                with fhe.tag(f"round-{j + 1}"):
                    if j < 16:
                        w[j] = chunks[j]
                    else:
                        w[j] = add(w[j - 16], s0(w[j - 15]), w[j - 7], s1(w[j - 2]))

                    w_i_plus_k_i = add(w[j], k_chunks[j])
                    state = state_update(state, w_i_plus_k_i)

            with fhe.tag("state-update"):
                h_chunks = [add(a, b) for a, b in zip(h_chunks, state)]

    with fhe.tag("result"):
        result = [[chunks[i] for i in range(chunks.size)] for chunks in h_chunks]
        return fhe.array(result)


def sha256_preprocess(text):
    l = text.shape[0] * 8
    k = (((448 - 1 - l) % 512) + 512) % 512

    zero_pad_width_in_bits = k
    pad_string = "1" + ("0" * zero_pad_width_in_bits) + np.binary_repr(l, width=64)

    total_size = len(pad_string) + l
    assert total_size % 512 == 0

    pad = [int(pad_string[i : i + 8], 2) for i in range(0, len(pad_string), 8)]
    padded = np.concatenate((text, pad))

    chunked_bytes = np.array([number_to_chunks(byte, size=8) for byte in padded])
    data = chunked_bytes.reshape(-1, NUMBER_OF_CHUNKS)

    return data


def chunks_to_hexdigest(chunked_numbers):
    hexes = [hex(chunks_to_number(chunks))[2:] for chunks in chunked_numbers]
    return "".join([("0" * (8 - len(h))) + h for h in hexes])


class HomomorphicSHA:
    input_size: int
    number_of_rounds: int
    circuit: fhe.Circuit

    def __init__(self, input_size, number_of_rounds=64) -> None:
        assert 0 <= number_of_rounds <= 64

        self.input_size = input_size
        self.number_of_rounds = number_of_rounds

        compiler = fhe.Compiler(
            lambda data: sha256(data, self.number_of_rounds), {"data": "encrypted"}
        )
        inputset = [
            sha256_preprocess(np.random.randint(0, 2**8, size=(self.input_size,)))
            for _ in range(100)
        ]

        self.circuit = compiler.compile(
            inputset=inputset,
            configuration=fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
                dataflow_parallelize=True,
                show_progress=True,
                progress_tag=True,
            ),
        )

    def run(self, data):
        if len(data) != self.input_size:
            message = f"Expected input to be {self.input_size} bytes, but it's {len(data)} bytes"
            raise ValueError(message)

        return self.circuit.encrypt_run_decrypt(sha256_preprocess(data))


def main():
    sample = np.random.randint(0, 2**8, size=(150,), dtype=np.uint8)
    preprocessed_sample = sha256_preprocess(np.frombuffer(sample, dtype=np.uint8))
    actual_result = chunks_to_hexdigest(sha256(preprocessed_sample))

    hasher = hashlib.sha256()
    hasher.update(bytes(sample))
    expected_result = hasher.hexdigest()

    print("Our SHA256:", actual_result)
    print("    SHA256:", expected_result)
    print("     Match:", actual_result == expected_result)

    for input_size in [50, 100, 150]:
        for number_of_rounds in [1, 2, 3, 10, 16, 32, 64]:
            print()

            title = (
                f"chunk_width={CHUNK_WIDTH}, "
                f"number_of_rounds={number_of_rounds}, "
                f"input_size={input_size}"
            )
            print(title)
            print("-" * len(title))

            print("Compiling...")
            start = time.time()
            sha = HomomorphicSHA(input_size, number_of_rounds)
            end = time.time()
            print(f"(took {end - start:.3f} seconds)")

            sha.circuit.server.save(
                f"{CHUNK_WIDTH}-bit-chunks/"
                f"{number_of_rounds}-rounds/"
                f"{input_size}-byte-input/"
                f"server.zip"
            )
            sha.circuit.client.save(
                f"{CHUNK_WIDTH}-bit-chunks/"
                f"{number_of_rounds}-rounds/"
                f"{input_size}-byte-input/"
                f"client.zip"
            )

            print()

            print(f"complexity: {sha.circuit.complexity}")
            print(f"p_error: {sha.circuit.p_error}")
            print(f"global_p_error: {sha.circuit.global_p_error}")
            print(f"total nodes: {sha.circuit.graph.node_count()}")
            print(f"nodes: {sha.circuit.graph.node_count_by_name()}")

            print()

            print("Generating keys...")
            start = time.time()
            sha.circuit.keygen()
            end = time.time()
            print(f"(took {end - start:.3f} seconds)")

            print()

            sample = np.random.randint(0, 2**8, size=(input_size,), dtype=np.uint8)
            data = np.frombuffer(sample, dtype=np.uint8)

            preprocessed = sha256_preprocess(data)
            plain_result = sha256(preprocessed, number_of_rounds)

            print("Running...")
            start = time.time()
            homomorphic_result = sha.run(data)
            end = time.time()
            print(f"(took {end - start:.3f} seconds)")

            print()

            homomorphic_hexdigest = chunks_to_hexdigest(homomorphic_result)
            plain_hexdigest = chunks_to_hexdigest(plain_result)

            print("Encrypted evaluation:", homomorphic_hexdigest)
            print("    Plain evaluation:", plain_hexdigest)
            print("               Match:", homomorphic_hexdigest == plain_hexdigest)

            print()


if __name__ == "__main__":
    main()

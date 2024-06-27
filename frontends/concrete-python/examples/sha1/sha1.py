# Forked and modified from https://github.com/ajalt/python-sha1, whose license was
#
#        The MIT License (MIT)
#
#        Copyright (c) 2013-2015 AJ Alt
#
#        Permission is hereby granted, free of charge, to any person obtaining a copy of
#        this software and associated documentation files (the "Software"), to deal in
#        the Software without restriction, including without limitation the rights to
#        use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#        the Software, and to permit persons to whom the Software is furnished to do so,
#        subject to the following conditions:
#
#        The above copyright notice and this permission notice shall be included in all
#        copies or substantial portions of the Software.
#
#        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#        FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#        COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#        IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#        CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import io
import random
import string
import struct
import time
from hashlib import sha1 as hashlib_sha1

import numpy as np

from concrete import fhe


def _left_rotate(n, b):
    """Left rotate a 32-bit integer n by b bits."""
    return ((n << b) | (n >> (32 - b))) & 0xFFFFFFFF


def split(b):
    """Splitting into bits."""
    ans = []
    for _ in range(32):
        ans += [b % 2]
        b = b // 2

    return np.array(ans, dtype=np.int8)


def unsplit(bits):
    """Unsplitting from bits to uint32."""
    ans = 0
    for i in range(32):
        ans *= 2
        ans += bits[31 - i]

    return ans


def get_random_string(length):
    """Return a random string."""
    if length == 0:
        return ""

    # ruff: noqa:S311
    result_str = "".join(random.choice(string.ascii_letters) for i in range(length))
    return result_str


def add_chunked_number(x, y):
    result = fhe.zeros((32,))
    carry = 0

    addition = x + y
    for i in range(32):
        addition_and_carry = addition[i] + carry
        carry = addition_and_carry >> 1
        result[i] = addition_and_carry - (carry * 2)

    return result


# FHE functions
@fhe.module()
class MyModule:
    @staticmethod
    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def xor3(x, y, z):
        return x ^ y ^ z

    @staticmethod
    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def iftern(x, y, z):
        return z ^ (x & (y ^ z))

    @staticmethod
    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def maj(x, y, z):
        return (x & y) | (z & (x | y))

    @staticmethod
    @fhe.function({"x": "encrypted"})
    def rotate30(x):
        ans = fhe.zeros((32,))
        ans[30:32] = x[0:2]
        ans[0:30] = x[2:32]
        return ans

    @staticmethod
    @fhe.function({"x": "encrypted"})
    def rotate5(x):
        ans = fhe.zeros((32,))
        ans[5:32] = x[0:27]
        ans[0:5] = x[27:32]
        return ans

    @staticmethod
    @fhe.function({"x": "encrypted", "y": "encrypted"})
    def add2(x, y):
        return fhe.bits(add_chunked_number(x, y))[0]

    @staticmethod
    @fhe.function(
        {"x": "encrypted", "y": "encrypted", "u": "encrypted", "v": "encrypted", "w": "encrypted"}
    )
    def add5(x, y, u, v, w):
        result = add_chunked_number(x, y)
        result = add_chunked_number(result, u)
        result = add_chunked_number(result, v)
        result = add_chunked_number(result, w)

        return fhe.bits(result)[0]


# Compilation of the FHE functions
size_of_inputsets = 1000
inputset1 = [(np.random.randint(2, size=(32,)),) for _ in range(size_of_inputsets)]
inputset2 = [
    (
        np.random.randint(2, size=(32,)),
        np.random.randint(2, size=(32,)),
    )
    for _ in range(size_of_inputsets)
]
inputset3 = [
    (
        np.random.randint(2, size=(32,)),
        np.random.randint(2, size=(32,)),
        np.random.randint(2, size=(32,)),
    )
    for _ in range(size_of_inputsets)
]
inputset5 = [
    (
        np.random.randint(2, size=(32,)),
        np.random.randint(2, size=(32,)),
        np.random.randint(2, size=(32,)),
        np.random.randint(2, size=(32,)),
        np.random.randint(2, size=(32,)),
    )
    for _ in range(size_of_inputsets)
]
# FIXME: remove the mypy and ruff exceptions once
# https://github.com/zama-ai/concrete-internal/issues/721 is fixed
# pylint: disable-next=no-member
my_module = MyModule.compile(  # type: ignore
    {
        "xor3": inputset3,
        "iftern": inputset3,
        "maj": inputset3,
        "rotate30": inputset1,
        "rotate5": inputset1,
        "add2": inputset2,
        "add5": inputset5,
    },
    show_mlir=False,
    bitwise_strategy_preference=fhe.BitwiseStrategy.ONE_TLU_PROMOTED,
    multivariate_strategy_preference=fhe.MultivariateStrategy.PROMOTED,
    p_error=10**-8,
)


# Split and encrypt on the client side
def message_schedule_and_split_and_encrypt(chunk):

    assert len(chunk) == 64

    w = [0] * 80

    # Break chunk into sixteen 4-byte big-endian words w[i]
    for i in range(16):
        w[i] = struct.unpack(b">I", chunk[i * 4 : i * 4 + 4])[0]

    # Extend the sixteen 4-byte words into eighty 4-byte words
    for i in range(16, 80):
        w[i] = _left_rotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1)

    # Then split and encrypt
    wsplit_enc = [0] * 80

    for i in range(80):
        wsplit_enc[i] = my_module.rotate5.encrypt(split(w[i]))

    return wsplit_enc


# Perform SHA computation server side, completely in FHE
def _process_encrypted_chunk_server_side(
    wsplit_enc, h0split_enc, h1split_enc, h2split_enc, h3split_enc, h4split_enc
):
    """Process a chunk of data and return the new digest variables."""

    # Initialize hash value for this chunk
    asplit_enc = h0split_enc
    bsplit_enc = h1split_enc
    csplit_enc = h2split_enc
    dsplit_enc = h3split_enc
    esplit_enc = h4split_enc

    for i in range(80):
        if 0 <= i <= 19:

            # Do f = d ^ (b & (c ^ d))
            fsplit_enc = my_module.iftern.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0x5A827999)
        elif 20 <= i <= 39:

            # Do f = b ^ c ^ d
            fsplit_enc = my_module.xor3.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0x6ED9EBA1)
        elif 40 <= i <= 59:

            # Do f = (b & c) | (b & d) | (c & d)
            fsplit_enc = my_module.maj.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0x8F1BBCDC)
        elif 60 <= i <= 79:

            # Do f = b ^ c ^ d
            fsplit_enc = my_module.xor3.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0xCA62C1D6)

        # Do arot5 = _left_rotate(a, 5)
        arot5split_enc = my_module.rotate5.run(asplit_enc)

        # Do arot5 + f + e + k + w[i]
        ssplit_enc = my_module.add5.run(
            arot5split_enc,
            fsplit_enc,
            esplit_enc,
            wsplit_enc[i],
            my_module.rotate5.encrypt(ksplit),  # BCM: later remove the encryption on k
        )

        # Final update of the a, b, c, d and e registers
        newasplit_enc = ssplit_enc

        esplit_enc = dsplit_enc
        dsplit_enc = csplit_enc

        # Do c = _left_rotate(b, 30)
        csplit_enc = my_module.rotate30.run(bsplit_enc)

        bsplit_enc = asplit_enc
        asplit_enc = newasplit_enc

    # Add this chunk's hash to result so far
    h0split_enc = my_module.add2.run(h0split_enc, asplit_enc)
    h1split_enc = my_module.add2.run(h1split_enc, bsplit_enc)
    h2split_enc = my_module.add2.run(h2split_enc, csplit_enc)
    h3split_enc = my_module.add2.run(h3split_enc, dsplit_enc)
    h4split_enc = my_module.add2.run(h4split_enc, esplit_enc)

    return h0split_enc, h1split_enc, h2split_enc, h3split_enc, h4split_enc


class Sha1Hash:
    """A class that mimics that hashlib api and implements the SHA-1 algorithm."""

    name = "python-sha1"
    digest_size = 20
    block_size = 64

    def __init__(self):
        # Initial digest variables
        h0, h1, h2, h3, h4 = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0)

        # Split
        h0split = split(h0)
        h1split = split(h1)
        h2split = split(h2)
        h3split = split(h3)
        h4split = split(h4)

        # Encrypt
        h0split_enc = my_module.rotate5.encrypt(h0split)
        h1split_enc = my_module.rotate5.encrypt(h1split)
        h2split_enc = my_module.rotate5.encrypt(h2split)
        h3split_enc = my_module.rotate5.encrypt(h3split)
        h4split_enc = my_module.rotate5.encrypt(h4split)

        self._hsplit_enc = (h0split_enc, h1split_enc, h2split_enc, h3split_enc, h4split_enc)

        # bytes object with 0 <= len < 64 used to store the end of the message
        # if the message length is not congruent to 64
        self._unprocessed = b""
        # Length in bytes of all data that has been processed so far
        self._message_byte_length = 0

    def update(self, arg):
        """Update the current digest.

        This may be called repeatedly, even after calling digest or hexdigest.

        Arguments:
            arg: bytes, bytearray, or BytesIO object to read from.
        """
        if isinstance(arg, (bytes, bytearray)):
            arg = io.BytesIO(arg)

        # Try to build a chunk out of the unprocessed data, if any
        chunk = self._unprocessed + arg.read(64 - len(self._unprocessed))

        # Read the rest of the data, 64 bytes at a time
        while len(chunk) == 64:

            wsplit_enc = message_schedule_and_split_and_encrypt(chunk)
            self._hsplit_enc = _process_encrypted_chunk_server_side(wsplit_enc, *self._hsplit_enc)
            self._message_byte_length += 64
            chunk = arg.read(64)

        self._unprocessed = chunk
        return self

    def digest(self):
        """Produce the final hash value (big-endian) as a bytes object"""
        return b"".join(struct.pack(b">I", h) for h in self._produce_digest())

    def hexdigest(self):
        """Produce the final hash value (big-endian) as a hex string"""
        local_digest = self._produce_digest()
        return (
            f"{local_digest[0]:08x}{local_digest[1]:08x}{local_digest[2]:08x}"
            + f"{local_digest[3]:08x}{local_digest[4]:08x}"
        )

    def _produce_digest(self):
        """Return finalized digest variables for the data processed so far."""
        # Pre-processing:
        message = self._unprocessed
        message_byte_length = self._message_byte_length + len(message)

        # append the bit '1' to the message
        message += b"\x80"

        # append 0 <= k < 512 bits '0', so that the resulting message length (in bytes)
        # is congruent to 56 (mod 64)
        message += b"\x00" * ((56 - (message_byte_length + 1) % 64) % 64)

        # append length of message (before pre-processing), in bits, as 64-bit big-endian integer
        message_bit_length = message_byte_length * 8
        message += struct.pack(b">Q", message_bit_length)

        # Process the final chunk
        # At this point, the length of the message is either 64 or 128 bytes.
        wsplit_enc = message_schedule_and_split_and_encrypt(message[:64])
        hsplit_enc = _process_encrypted_chunk_server_side(wsplit_enc, *self._hsplit_enc)

        if len(message) != 64:

            wsplit_enc = message_schedule_and_split_and_encrypt(message[64:])
            hsplit_enc = _process_encrypted_chunk_server_side(wsplit_enc, *hsplit_enc)

        # Decrypt
        h0split = my_module.rotate5.decrypt(hsplit_enc[0])
        h1split = my_module.rotate5.decrypt(hsplit_enc[1])
        h2split = my_module.rotate5.decrypt(hsplit_enc[2])
        h3split = my_module.rotate5.decrypt(hsplit_enc[3])
        h4split = my_module.rotate5.decrypt(hsplit_enc[4])

        # Unsplit
        h0 = unsplit(h0split)
        h1 = unsplit(h1split)
        h2 = unsplit(h2split)
        h3 = unsplit(h3split)
        h4 = unsplit(h4split)

        return h0, h1, h2, h3, h4


def sha1(local_data):
    """SHA-1 Hashing Function

    A custom SHA-1 hashing function implemented entirely in Python.

    Arguments:
        local_data: A bytes or BytesIO object containing the input message to hash.

    Returns:
        A hex SHA-1 digest of the input message.
    """
    return Sha1Hash().update(local_data).hexdigest()


def print_timed_sha1(local_data):
    time_begin = time.time()
    ans = sha1(local_data)
    print(f"sha1-digest: {ans}")
    print(f"computed in: {time.time() - time_begin:2f} seconds")
    return ans


if __name__ == "__main__":
    # Imports required for command line parsing. No need for these elsewhere
    import argparse
    import os
    import sys

    # Parse the incoming arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="*", help="input file or message to hash")
    parser.add_argument("--autotest", action="store_true", help="autotest")
    args = parser.parse_args()

    if args.autotest:

        filename = "tmp_sha1_test_file.txt"

        # Checking random patterns
        for _ in range(20):
            string_length = 4

            # Take a random string
            hash_input = get_random_string(string_length)

            print(f"Checking SHA1({hash_input}) for an input length {string_length}")

            # Hash it with hashlib_sha1
            # ruff: noqa:S324
            h = hashlib_sha1()
            h.update(bytes(hash_input, encoding="utf-8"))
            expected_ans = h.hexdigest()

            # Hash it in FHE
            with open(filename, "w", encoding="utf-8") as file:
                file.write(f"{hash_input}")

            with open(filename, "rb") as data:
                # Show the final digest
                actual_ans = print_timed_sha1(data)

            # And compare
            assert (
                actual_ans == expected_ans
            ), f"Wrong computation: {actual_ans} vs expected {expected_ans} for input {hash_input}"

        # Checking a few patterns
        for hash_input, expected_ans in [
            ("", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
            (
                "The quick brown fox jumps over the lazy dog",
                "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12",
            ),
        ]:

            with open(filename, "w", encoding="utf-8") as file:
                file.write(f"{hash_input}")

            print(f"Checking SHA1({hash_input})")

            with open(filename, "rb") as data:
                # Show the final digest
                actual_ans = print_timed_sha1(data)

            assert (
                actual_ans == expected_ans
            ), f"Wrong computation: {actual_ans} vs expected {expected_ans}"

        sys.exit(0)

    if len(args.input) == 0:
        # No argument given, assume message comes from standard input
        try:
            # sys.stdin is opened in text mode, which can change line endings,
            # leading to incorrect results. Detach fixes this issue, but it's
            # new in Python 3.1
            data = sys.stdin.detach()  # type: ignore

        except AttributeError:
            # Linux and OSX both use \n line endings, so only windows is a
            # problem.
            if sys.platform == "win32":
                import msvcrt

                msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
            data = sys.stdin  # type: ignore

        # Output to console
        print_timed_sha1(data)

    else:
        # Loop through arguments list
        for argument in args.input:
            if os.path.isfile(argument):
                # An argument is given and it's a valid file. Read it
                with open(filename, "rb") as data:

                    # Show the final digest
                    print_timed_sha1(data)

            else:
                print("Error, could not find " + argument + " file.")

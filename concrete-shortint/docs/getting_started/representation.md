# How Shortint are represented

In `concrete-shortint`, the encrypted data is stored in an LWE ciphertext.

Conceptually, the message stored in an LWE ciphertext, is divided into 
a **carry buffer** and a **message buffer**.

![](ciphertext-representation.svg)

The message buffer is the space where the actual message is 
stored. This represents the modulus of the input messages 
(denoted by `MessageModulus` in the code) 
When doing computations on a ciphertext, the encrypted message can overflow the message 
modulus: the exceeding information is stored in 
the carry buffer. The size of the carry buffer is defined by another modulus, called 
`CarryModulus`.

Together, the message modulus and the carry modulus form the plaintext space that is
available in a ciphertext. This space cannot be overflowed, otherwise the computation may result 
in incorrect outputs. 

In order to ensure the computation correctness, we keep track of the maximum value encrypted in a 
ciphertext
via an associated attribute called the **degree**. When the degree reaches a defined 
threshold, the carry buffer may be emptied to resume safely the computations. 
Therefore, in `concrete-shortint` the carry modulus is mainly considered as a means to do more 
computations.

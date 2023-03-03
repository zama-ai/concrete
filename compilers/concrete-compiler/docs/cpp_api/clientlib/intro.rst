Description
========

ClientLambda represents a FHE function on the client side.
ServerLambda represents a FHE function on the server side.

These object read/write on istreams/ostreams.

Implementing a client/server consists in connecting the two, by connecting to actual streams.

Example on client side:

.. literalinclude:: client_example.cpp
    :linenos:
    :language: bash
..
  For some reason cpp does not work well here.

Example on server side:

.. literalinclude:: server_example.cpp
    :linenos:
    :language: bash

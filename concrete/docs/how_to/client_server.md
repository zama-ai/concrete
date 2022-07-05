# Client Server

In the [concrete-example-client-server](https://github.com/zama-ai/concrete-example-client-server) GitHub repository, we have a more complete example project of an application that is composed of a client and a server.

This example demonstrates the use of the following aspects:

* Building Client Server architecture
* Communicating via `TcpStream`
* Using serialization to exchange encrypted data
* Using serialization to save a client key locally and save processing time
* Creating a generic function for use with different `Fhe` types
* Simple multithreading on the server side to handle multiple clients at the same time

## Server explanation

Communication is achieved via a tcp connection. The server is the listener, so it creates a `TcpListener` that listens for incoming connections on `localhost` port `8080`.

When a client initiates a connection, the main server thread calls the `handle_client` function in a new thread (and also moves the tcp connection to this new thread).

If we did not create another thread, a client connected to the server would have to wait for the previous client to finish and end its connection before proceeding.

```rust
fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080")?;
    println!("Server is listening");

    // accept connections and process them sequentially
    for stream in listener.incoming() {
        println!("A client initiated connection");
        std::thread::spawn(move || {
            handle_client(stream?)
        });
    }
    Ok(())
}
```

The first thing that the server does is receive and deserialize the `ServerKey` sent by the client. Then, it immediately calls `set_server_key`.

```rust
fn handle_client(mut stream: TcpStream) -> std::io::Result<()> {
    println!("[Server] <---- [Client]: Receiving server keys from client");
    let server_keys: ServerKey = bincode::deserialize_from(&mut stream).unwrap();
    set_server_key(server_keys);
    
    /// ...
    /// 
    Ok(())
}
```

Once the key is set, the function starts an infinite `loop`.

The first step of the loop is the server receiving a "token" sent from the client, to know if the client wishes to stop the connection.

```rust
fn handle_client(mut stream: TcpStream) -> std::io::Result<()> {
    /// ...
    loop {
        let choice = stream.read_u8()?;
        if choice == 0 {
            println!("[Server] <---- [Client]: User said good bye");
            break;
        }
        /// ...
    }
    Ok(())
}
```

The second step is rather simple: The server expects the client to send 3 `FheUint3`s; the server deserializes them and performs a `fhe_computation` on them; and, finally, the server serializes the results and sends them to the client.

```rust
fn handle_client(mut stream: TcpStream) -> std::io::Result<()> {
    // ...
    loop {
        // ...
        {
            println!("[Server] <---- [Client]: Receiving a, b, c");
            let a: FheUint3 = bincode::deserialize_from(&mut stream).unwrap();
            let b: FheUint3 = bincode::deserialize_from(&mut stream).unwrap();
            let c: FheUint3 = bincode::deserialize_from(&mut stream).unwrap();

            print!("Computing...");
            let result = fhe_computation(&a, &b, &c);
            println!("done.");
            println!("[Server] ----> [Client]: Sending Result");
            bincode::serialize_into(&mut stream, &result).unwrap();
        }
        // ...
    }
    Ok(())
}
```

The last step is similar to the previous one. The difference is that the server expects and uses `FheUint16`.

```rust
fn handle_client(mut stream: TcpStream) -> std::io::Result<()> {
    // ...
    loop {
        // ...
        {
            println!("[Server] <---- [Client]: Receiving a, b, c");
            let a: FheUint16 = bincode::deserialize_from(&mut stream).unwrap();
            let b: FheUint16 = bincode::deserialize_from(&mut stream).unwrap();
            let c: FheUint16 = bincode::deserialize_from(&mut stream).unwrap();

            print!("Computing...");
            let result = fhe_computation(&a, &b, &c);
            println!("done.");
            println!("[Server] ----> [Client]: Sending Result");
            bincode::serialize_into(&mut stream, &result).unwrap();
        }
    }
    Ok(())
}
```

The `fhe_computation` is a generic function defined as:

```rust
fn fhe_computation<'a, T>(a: &'a T, b: &'a T, c: &'a T) -> T
where &'a T: Add<&'a T, Output=T>,
     T: Mul<&'a T, Output=T>
{
    (a + b) * c
}
```

## Client explanation

The client code has a more code, as it does a bit more than just communicating with the server. It interacts via the standard input with a user and manages the keys, as well.

First, the client generates the `ClientKey` and the `ServerKey`. It uses the custom function `key_gen` to do so. The `key_gen` function's goal is to save the file on disk and reuse the saved keys to avoid regenerating them each time the client process starts, saving a lot of time.

```rust
use crate::details::{ask_for_exit, fhe16_from_stin, fhe3_from_stin, key_gen};

fn main() -> Result<(), Box<dyn Error>> {
    let ( client_keys, mut server_keys) = key_gen()?;

    // ...
    Ok(())
}
```

```rust
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, stdin};
use std::path::Path;
use concrete::{ClientKey, ConfigBuilder, FheUint3Parameters, ServerKey};
use concrete::prelude::*;

const CLIENT_KEY_FILE_PATH: &'static str = "client_key.bin";
const SERVER_KEY_FILE_PATH: &'static str = "server_key.bin";

pub fn key_gen() -> Result<(ClientKey, ServerKey), Box<dyn Error>> {
    let client_key_path = Path::new(CLIENT_KEY_FILE_PATH);

    let client_keys: ClientKey =
        if client_key_path.exists() {
            println!("Reading client keys from {}", CLIENT_KEY_FILE_PATH);
            let mut file = BufReader::new(File::open(client_key_path)?);
            bincode::deserialize_from(file)?
        } else {
            println!("No {} found, generating new keys and saving them", CLIENT_KEY_FILE_PATH);
            let config = ConfigBuilder::all_disabled().enable_default_uint3().enable_default_uint16().build();
            let k = ClientKey::generate(config);
            let file = BufWriter::new(File::create(client_key_path)?);
            bincode::serialize_into(file, &k)?;

            k
        };

    let server_key_path = Path::new(SERVER_KEY_FILE_PATH);
    let server_keys: ServerKey = if server_key_path.exists() {
        println!("Reading server keys from {}", CLIENT_KEY_FILE_PATH);
        let mut file = BufReader::new(File::open(server_key_path)?);
        bincode::deserialize_from(file).unwrap()
    } else {
        println!("No {} found, generating new keys and saving them", SERVER_KEY_FILE_PATH);
        let k = client_keys.generate_server_key();
        let file = BufWriter::new(File::create(server_key_path)?);
        bincode::serialize_into(file, &k).unwrap();

        k
    };

    Ok((client_keys, server_keys))
}
```

Next, the tcp connection is initiated, and the `ServerKey` is sent to the server.

```rust
fn main() -> Result<(), Box<dyn Error>> {
    let ( client_keys, mut server_keys) = key_gen()?;
    
    println!("[Client] ----> [Server]: Connecting to server");
    let mut stream = TcpStream::connect("127.0.0.1:8080")?;

    println!("[Client] ----> [Server]: Sending Bootstrap Keys to server");
    bincode::serialize_into(&mut stream, &server_keys)?;

    Ok(())
}
```

Once the key was successfully sent, the client does the same thing as the server: it enters an infinite `loop`.

The first step of the loop is to send the value `1` to the server to tell it we want to continue.

```rust
fn main() -> Result<(), Box<dyn Error>> {
    // ...
    loop {
        stream.write_u8(1)?;

        // ...
    }

    Ok(())
}
```

Then, the client reads from the standard input 3 numbers that must fit in 3 bits. These numbers are then encrypted, serialized, and sent to the server.

Next, the client reads the result returned by the server and deserializes, decrypts, and prints the result on the standard output.

This is done again, but this time for numbers that can fit into 16 bits.

```rust
fn main() -> Result<(), Box<dyn Error>> {
    // ...
    loop {
        // ...
        {
            let a = fhe3_from_stin(&client_keys);
            let b = fhe3_from_stin(&client_keys);
            let c = fhe3_from_stin(&client_keys);

            println!("[Client] ----> [Server]: Sending a, b, c");
            bincode::serialize_into(&mut stream, &a)?;
            bincode::serialize_into(&mut stream, &b)?;
            bincode::serialize_into(&mut stream, &c)?;

            println!("[Client] <---- [Server]: Receiving result");
            let result: FheUint3 = bincode::deserialize_from(&mut stream).unwrap();

            let clear_result = result.decrypt(&client_keys);
            println!("The result is: {}", clear_result);
        }

        {
            let a = fhe16_from_stin(&client_keys);
            let b = fhe16_from_stin(&client_keys);
            let c = fhe16_from_stin(&client_keys);

            println!("[Client] ----> [Server]: Sending a, b, c");
            bincode::serialize_into(&mut stream, &a)?;
            bincode::serialize_into(&mut stream, &b)?;
            bincode::serialize_into(&mut stream, &c)?;

            println!("[Client] <---- [Server]: Receiving result");
            let result: FheUint16 = bincode::deserialize_from(&mut stream).unwrap();
            let clear_result: u16 = result.decrypt(&client_keys);

            println!("The result is: {}", clear_result);
        }
        // ...
    }

    Ok(())
}
```

Finally, the clients ask the user if it wishes to perform further computations. If that is the case, the client will send the value `0` to the server, letting it know that the connection can end.

```rust
fn main() -> Result<(), Box<dyn Error>> {
    // ...
    loop {
        // ...

        let should_exit = ask_for_exit();
        if should_exit {
            stream.write_u8(0)?;
            break
        }
    }

    Ok(())
}
```

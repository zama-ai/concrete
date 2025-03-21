// tested with nodejs v22.14.0

const fs = require('fs');
const cr = require("../pkg/concrete_rust")

let keyset_info_path = "./ks_info.capnp";
let output_keyset_path = "ks_no_bsk.capnp";

let msgChannel = new MessageChannel();
msgChannel.port1.onmessage = (ev) => {
    fs.appendFileSync('chunks.bytes', ev.data);
}
msgChannel.port1.on("close", () => {
    console.log("channel closed");
});

let keyset_info_buffer = fs.readFileSync(keyset_info_path);

let keyset_buffer_no_bsk = cr.generate_keyset_wasm(keyset_info_buffer, true, {}, true, {}, BigInt(128), BigInt(128));

let sk_in = cr.get_lwe_secret_key_from_keyset_wasm(keyset_buffer_no_bsk, 5);
let sk_out = cr.get_lwe_secret_key_from_keyset_wasm(keyset_buffer_no_bsk, 4);
cr.chunked_bsk_keygen(keyset_info_buffer, sk_in, sk_out, 2, BigInt(0), 8, msgChannel.port2)
    .then(returned_value => {
        console.log(returned_value);
    })
    .catch(error => {
        console.error("Error:", error);
    });
fs.writeFileSync(output_keyset_path, keyset_buffer_no_bsk)


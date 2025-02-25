// tested with nodejs v22.14.0

const fs = require('fs');
const cr = require("../pkg/concrete_rust")

let keyset_info_path = "./ks_info.capnp";
let output_keyset_path = "ks.capnp";

let keyset_info_buffer = fs.readFileSync(keyset_info_path);

let keyset_buffer = cr.generate_keyset_wasm(keyset_info_buffer, BigInt(0), BigInt(0), BigInt(0), BigInt(0))

fs.writeFileSync(output_keyset_path, keyset_buffer)


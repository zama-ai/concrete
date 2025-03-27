// tested with nodejs v22.14.0

const fs = require('fs');
const cr = require("../pkg/concrete_rust");
const { exit } = require('process');

let keyset_info_path = "./ks_info.capnp";
let output_keyset_path = "ks_no_bsk.capnp";

const forceDelete = process.argv.includes("--rm");

if (fs.existsSync(output_keyset_path)) {
    if (!forceDelete) {
        console.log(`File ${output_keyset_path} exists, delete it manually or use '--rm' option`);
        exit(1);
    }
    fs.rmSync(output_keyset_path);
}

function createMessageChannel(filePath) {
    let msgChannel = new MessageChannel();
    if (fs.existsSync(filePath)) {
        if (!forceDelete) {
            console.log(`File ${filePath} exists, delete it manually or use '--rm' option`);
            exit(1);
        }
        fs.rmSync(filePath);
    }
    msgChannel.port1.onmessage = (ev) => {
        fs.appendFileSync(filePath, ev.data);
    };
    msgChannel.port1.on("close", () => {
        console.log("channel closed");
    });
    return msgChannel;
}

let keyset_info_buffer = fs.readFileSync(keyset_info_path);
let keyset_info_json = cr.explain_keyset_info(keyset_info_buffer);
console.log("keyset_info explained:");
console.dir(keyset_info_json, { depth: null });

let keyset_buffer_no_bsk_no_ksk = cr.generate_keyset_wasm(keyset_info_buffer, true, {}, true, {}, BigInt(128), BigInt(128));

(async () => {
    for (let bsk of keyset_info_json.lwe_bootstrap_keys) {
        let msgChannel = createMessageChannel(`bsk_chunked_${bsk.id}.bytes`);
        let sk_in = cr.get_lwe_secret_key_from_keyset_wasm(keyset_buffer_no_bsk_no_ksk, bsk.params.input_id);
        let sk_out = cr.get_lwe_secret_key_from_keyset_wasm(keyset_buffer_no_bsk_no_ksk, bsk.params.output_id);
        try {
            let returned_value = await cr.chunked_bsk_keygen(
                keyset_info_buffer,
                sk_in,
                sk_out,
                bsk.id,
                BigInt(0),
                8,
                msgChannel.port2
            );
            console.log(`Bootstrap key ${bsk.id} generated successfully (returned value: ${returned_value})`);
        } catch (error) {
            console.error(`Error generating bootstrap key ${bsk.id}:`, error);
        }
    }
})();

// TODO:
// let keyset_buffer_no_bsk = cr.generate_keyswitch_keys(keyset_buffer_no_bsk_no_ksk, ...);
// fs.writeFileSync(output_keyset_path, keyset_buffer_no_bsk)
// On server side, we should be able to assemble bsk keys first,
// then full keyset from keyset_buffer_no_bsk and those bsk keys
fs.writeFileSync(output_keyset_path, keyset_buffer_no_bsk_no_ksk)

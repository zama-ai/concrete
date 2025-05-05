import("./pkg/concrete_keygen.js").then((cr) => {
    window.cr = cr;
    cr.default();
});

async function handleKeygen(event) {
    event.preventDefault();

    const fileInput = document.getElementById('keyset-file');
    const chunkSizeInput = document.getElementById('chunk-size');
    const status = document.getElementById('status');
    const downloadLink = document.getElementById('download-link');

    if (!fileInput.files.length) {
        status.textContent = 'Please upload a keyset info file.';
        return;
    }

    const chunkSize = parseInt(chunkSizeInput.value, 10);
    if (!chunkSize || chunkSize <= 0) {
        status.textContent = 'Please enter a valid chunk size (>= 1).';
        return;
    }

    const file = fileInput.files[0];
    const keysetInfoBuffer = new Uint8Array(await file.arrayBuffer());

    status.textContent = 'Processing...';

    try {
        // Explain keyset info
        const keysetInfoJson = await cr.explain_keyset_info(keysetInfoBuffer);
        console.log('Keyset Info:', keysetInfoJson);

        // Generate client keyset
        const clientKeysetBuffer = cr.generate_client_keyset(keysetInfoBuffer, BigInt(128));
        console.log("Client keyset generation completed.")

        // Generate keyset without bootstrap keys
        const keysetBufferNoBsk = cr.generate_keyset(
            keysetInfoBuffer,
            true, // no_bsk
            {}, // ignore_bsk
            false, // no_ksk
            {}, // ignore_ksk
            BigInt(0),
            BigInt(0),
            clientKeysetBuffer
        );
        console.log("Keyset generation (without BSKs) completed.")

        // Create a zip file
        const zipFile = new zip.ZipWriter(new zip.BlobWriter("application/zip"), { bufferedWrite: true });
        zipFile.add('keyset_no_bsk.capnp', new zip.Uint8ArrayReader(keysetBufferNoBsk));
        zipFile.add('keyset_info.capnp', new zip.Uint8ArrayReader(keysetInfoBuffer));

        // Generate bootstrap keys in chunks
        for (const bsk of keysetInfoJson.lwe_bootstrap_keys) {
            const chunk_size = parseInt(document.getElementById('chunk-size').value, 10) || 8;
            var chunk_count = 0;
            const port = {
                postMessage: (data) => {
                    zipFile.add(`bsk_${bsk.id}_chunk_${chunk_count++}`, new zip.Uint8ArrayReader(data));
                },
                close: () => console.log(`Bootstrap key ${bsk.id} generation completed.`)
            };
            await cr.chunked_bsk_keygen(
                keysetInfoBuffer,
                cr.get_lwe_secret_key_from_client_keyset(clientKeysetBuffer, bsk.input_id),
                cr.get_lwe_secret_key_from_client_keyset(clientKeysetBuffer, bsk.output_id),
                bsk.id,
                BigInt(0),
                BigInt(0),
                chunk_size,
                port
            );
        }

        console.log("BSK chunks generation completed.")

        const url = URL.createObjectURL(await zipFile.close());

        downloadLink.href = url;
        downloadLink.download = 'chunked_keyset.zip';
        downloadLink.style.display = 'block';
        downloadLink.textContent = 'Download Result';

        status.textContent = 'Keyset generated successfully!';
    } catch (error) {
        console.error(error);
        status.textContent = `Error: ${error.message}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('keygen-form').addEventListener('submit', handleKeygen);
});

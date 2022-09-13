#![cfg(feature = "integers")]

use concrete::prelude::*;
use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint8};

#[test]
fn test_uint8() {
    let config = ConfigBuilder::all_disabled().enable_default_uint8().build();

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let a = FheUint8::encrypt(27, &client_key);
    let b = FheUint8::encrypt(100, &client_key);

    let c: u8 = (a + b).decrypt(&client_key);
    assert_eq!(c, 127);
}

mod dynamic {
    use concrete::prelude::*;
    use concrete::{
        generate_keys, set_server_key, ConfigBuilder, FheUint2Parameters, RadixParameters,
    };

    #[test]
    fn test_uint10() {
        let mut config = ConfigBuilder::all_disabled();
        let uint10_type = config.add_integer_type(RadixParameters {
            block_parameters: FheUint2Parameters::with_carry_2().into(),
            num_block: 5,
        });

        let (client_key, server_key) = generate_keys(config);

        set_server_key(server_key);

        let a = uint10_type.encrypt(127, &client_key);
        let b = uint10_type.encrypt(100, &client_key);

        let c: u64 = (a + b).decrypt(&client_key);
        assert_eq!(c, 227);
    }
}

#![feature(file_lock)]
#[allow(unused)]
use quote::quote;
use concrete::{compiler, protocol::ProgramInfo};
use configuration::Configuration;
use proc_macro::{
    TokenStream, {self},
};
use std::{fs::read_to_string, path::PathBuf};
use std::hash::{DefaultHasher, Hash, Hasher};
use syn::LitStr;

const CONCRETE_BUILD_DIR: &'static str = env!("CONCRETE_BUILD_DIR");
const PATH_STATIC_LIB: &'static str = "staticlib.a";
const PATH_PROGRAM_INFO: &'static str = "program_info.concrete.params.json";
const PATH_CIRCUIT: &'static str = "circuit.mlir";
const PATH_COMPOSITION_RULES: &'static str = "composition_rules.json";
const PATH_SIMULATED: &'static str = "is_simulated";
const PATH_CONFIGURATION: &'static str = "configuration.json";
const DEFAULT_GLOBAL_P_ERROR: Option<f64> = Some(0.00001);
const DEFAULT_P_ERROR: Option<f64> = None;

mod configuration;
mod fast_path_hasher;
mod unzip;
mod generation;

#[proc_macro]
pub fn from_concrete_python_export_zip(input: TokenStream) -> TokenStream {
    let pt: Result<LitStr, _> = syn::parse(input);

    let Ok(path_litteral) = pt else {
        panic!("Ununwraped input. Expected path string litteral.");
    };

    let path = PathBuf::from(path_litteral.value());
    if !path.is_relative() {
        panic!("Found absolute artifact path. Artifacts paths are resolved relative to the CARGO_MANIFEST_DIR directory.");
    }
    if std::env::var("CARGO_MANIFEST_DIR").is_err() {
        panic!("CARGO_MANIFEST_DIR environment variable not set (usually set by cargo). Something is wrong.");
    }
    let zip_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(path);
    if !zip_path.exists() {
        panic!("Input path must point to an existing export zip (relative to CARGO_MANIFEST_DIR): File {} not found.", zip_path.display());
    };
    let concrete_build_dir = PathBuf::from(CONCRETE_BUILD_DIR);

    let mut s = DefaultHasher::new();
    fast_path_hasher::FastPathHasher::from_pathbuf(&zip_path).hash(&mut s);
    let hash_val = s.finish();

    if !concrete_build_dir.exists() {
        let _ = std::fs::create_dir(&concrete_build_dir);
    }

    let lock_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(concrete_build_dir.join(format!("{hash_val}.lock")))
        .expect("Failed to open lock file.");

    lock_file.lock().expect("Failed to acquire lock on the lock file");

    let concrete_hash_dir = concrete_build_dir.join(format!("{hash_val}"));
    if !concrete_hash_dir.exists() {
        unzip::unzip(&zip_path, &concrete_hash_dir);
    }

    let circuit_path = concrete_hash_dir.join(PATH_CIRCUIT);
    let simulated_path = concrete_hash_dir.join(PATH_SIMULATED);
    let config_path = concrete_hash_dir.join(PATH_CONFIGURATION);
    let composition_rules_path = concrete_hash_dir.join(PATH_COMPOSITION_RULES);
    let output_lib_path = concrete_hash_dir.join(PATH_STATIC_LIB);

    if !output_lib_path.exists() {
        if !circuit_path.exists() {
            panic!("Missing `circuit.mlir` file in the export. Did you save your server with the `via_mlir` option ?");
        }
        let mlir = read_to_string(circuit_path).expect("Failed to read mlir sources to string");

        if !simulated_path.exists() {
            panic!("Missing `is_simulated` file in the export. Did you save your server with the `via_mlir` option ?");
        }
        let is_simulated = read_to_string(simulated_path)
            .expect("Failed to read is_simulated")
            .parse::<u8>()
            .expect("is_simulated can not be parsed as an u8 ...");

        if !config_path.exists() {
            panic!("Missing `configuration.json` file in the export. Did you save your server with the `via_mlir` option ?");
        }
        let configuration_string = read_to_string(config_path).expect("Failed to read configuration to string");
        let conf: Configuration = serde_json::from_str(configuration_string.as_str()).expect("Failed to deserialize configuration");

        if !composition_rules_path.exists() {
            panic!("Missing `composition_rules.json` file in the export. Did you save your server with the `via_mlir` option ?");
        }
        let composition_rules_string = read_to_string(composition_rules_path).expect("Failed to read composition rules to string");
        let composition_rules: Vec<serde_json::Value> =
            serde_json::from_str(composition_rules_string.as_str()).expect("Failed to deserialize composition rules");

        let mut opts = compiler::CompilationOptions::new();
        opts.pin_mut()
            .set_display_optimizer_choice(conf.show_optimizer.unwrap_or(false));
        opts.pin_mut().set_loop_parallelize(conf.loop_parallelize);
        opts.pin_mut()
            .set_dataflow_parallelize(conf.dataflow_parallelize);
        opts.pin_mut().set_auto_parallelize(conf.auto_parallelize);
        opts.pin_mut()
            .set_compress_evaluation_keys(conf.compress_evaluation_keys);
        opts.pin_mut()
            .set_compress_input_ciphertexts(conf.compress_input_ciphertexts);

        let global_p_error_is_set = conf.global_p_error.is_some();
        let p_error_is_set = conf.p_error.is_some();
        if global_p_error_is_set && p_error_is_set {
            opts.pin_mut()
                .set_global_p_error(conf.global_p_error.unwrap());
            opts.pin_mut().set_p_error(conf.p_error.unwrap());
        } else if global_p_error_is_set {
            opts.pin_mut()
                .set_global_p_error(conf.global_p_error.unwrap());
            opts.pin_mut().set_p_error(1.0)
        } else if p_error_is_set {
            opts.pin_mut().set_global_p_error(1.0);
            opts.pin_mut().set_p_error(conf.p_error.unwrap());
        } else {
            opts.pin_mut()
                .set_global_p_error(DEFAULT_GLOBAL_P_ERROR.unwrap_or(1.0));
            opts.pin_mut().set_p_error(DEFAULT_P_ERROR.unwrap_or(1.0));
        }

        match conf.parameter_selection_strategy {
            configuration::ParameterSelectionStrategy::V0 => {
                opts.pin_mut().set_optimizer_strategy(0)
            }
            configuration::ParameterSelectionStrategy::Mono => {
                opts.pin_mut().set_optimizer_strategy(1)
            }
            configuration::ParameterSelectionStrategy::Multi => {
                opts.pin_mut().set_optimizer_strategy(2)
            }
        }

        match conf.multi_parameter_strategy {
            configuration::MultiParameterStrategy::Precision => {
                opts.pin_mut().set_optimizer_multi_parameter_strategy(0)
            }
            configuration::MultiParameterStrategy::PrecisionAndNorm2 => {
                opts.pin_mut().set_optimizer_multi_parameter_strategy(1)
            }
        }

        opts.pin_mut().set_enable_tlu_fusing(conf.enable_tlu_fusing);
        opts.pin_mut().set_simulate(is_simulated != 0);
        opts.pin_mut()
            .set_enable_overflow_detection_in_simulation(conf.detect_overflow_in_simulation);
        opts.pin_mut().set_composable(conf.composable);
        opts.pin_mut()
            .set_range_restriction(&conf.range_restriction.map(|a| a.0).unwrap_or("".into()));
        opts.pin_mut()
            .set_keyset_restriction(&conf.keyset_restriction.map(|a| a.0).unwrap_or("".into()));
        match conf.security_level {
            configuration::SecurityLevel::Security128Bits => opts.pin_mut().set_security_level(128),
            configuration::SecurityLevel::Security132Bits => opts.pin_mut().set_security_level(132),
        }

        for rule in composition_rules {
            let serde_json::Value::Array(arr) = rule else {
                panic!()
            };
            let Some(serde_json::Value::Array(from)) = arr.get(0) else {
                panic!()
            };
            let Some(serde_json::Value::String(from_func)) = from.get(0) else {
                panic!()
            };
            let Some(serde_json::Value::Number(from_pos)) = from.get(1) else {
                panic!()
            };

            let Some(serde_json::Value::Array(to)) = arr.get(0) else {
                panic!()
            };
            let Some(serde_json::Value::String(to_func)) = to.get(0) else {
                panic!()
            };
            let Some(serde_json::Value::Number(to_pos)) = to.get(1) else {
                panic!()
            };
            opts.pin_mut().add_composition_rule(
                from_func,
                from_pos.as_u64().unwrap() as usize,
                to_func,
                to_pos.as_u64().unwrap() as usize,
            );
        }

        compiler::compile(
            mlir.as_str(),
            &opts,
            concrete_hash_dir.as_os_str().to_str().unwrap(),
        )
        .expect("Failed to compile sources");
    }

    let output_path = concrete_build_dir.join(format!("libconcrete-artifact-{hash_val}.a"));
    if !output_path.exists() {
        std::fs::copy(concrete_hash_dir.join(PATH_STATIC_LIB), output_path)
            .unwrap();
    }

    let concrete_program_info_path = concrete_hash_dir.join(PATH_PROGRAM_INFO);
    if !concrete_program_info_path.exists() {
        panic!("Missing `program_info.concrete.params.json` file after compilation. Something is wrong. Delete target folder and re-compile.");
    }
    let program_info: ProgramInfo = serde_json::from_reader(
        std::fs::File::open(concrete_program_info_path).unwrap(),
    )
    .unwrap();

    lock_file.unlock().unwrap();

    let lib_name = format!("concrete-artifact-{hash_val}");
    let unsafe_binding = generation::generate_unsafe_binding(&program_info);
    let infos = generation::generate_infos(&program_info);
    let keyset = generation::generate_keyset();
    let client = generation::generate_client(&program_info);
    let server = generation::generate_server(&program_info);

    quote! {
        #infos
        #keyset
        #client
        #server

        #[doc(hidden)]
        pub mod _binding {
            #[link(name = "ConcretelangRuntime", kind="dylib")]
            #[link(name = #lib_name, kind="static")]
            #unsafe_binding
        }
    }
    .into()
}

use concrete_sys::*;
use configuration::Configuration;
use proc_macro::TokenStream;
use proc_macro::{self};
use quote::quote;
use std::fs::read_to_string;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use syn::LitStr;
use zip;

const CONCRETE_BUILD_DIR: &'static str = env!("CONCRETE_SYS_BUILD_DIR");
const PATH_STATIC_LIB: &'static str = "staticlib.a";
const PATH_PROGRAM_INFO: &'static str = "program_info.concrete.params.json";
const PATH_CIRCUIT: &'static str = "circuit.mlir";
const PATH_COMPOSITION_RULES: &'static str = "composition_rules.json";
const PATH_SIMULATED: &'static str = "is_simulated";
const PATH_CONFIGURATION: &'static str = "configuration.json";
const DEFAULT_GLOBAL_P_ERROR: Option<f64> = Some(0.00001);
const DEFAULT_P_ERROR: Option<f64> = None;

mod configuration;

struct FastPathHasher {
    path: PathBuf,
    ctime: i64,
    mtime: i64,
}

impl FastPathHasher {
    fn from_pathbuf(path: &PathBuf) -> FastPathHasher {
        let path = path.canonicalize().unwrap();
        let metadata = path.metadata().unwrap();
        FastPathHasher {
            ctime: metadata.ctime(),
            mtime: metadata.mtime(),
            path,
        }
    }
}

impl Hash for FastPathHasher {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.ctime.hash(state);
        self.mtime.hash(state);
    }
}

fn unzip(zip_path: &Path, to: &Path) {
    let file = std::fs::File::open(zip_path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        let outpath = match file.enclosed_name() {
            Some(path) => to.join(path),
            None => continue,
        };
        if file.is_dir() {
            std::fs::create_dir_all(&outpath).unwrap();
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(p).unwrap();
                }
            }
            let mut outfile = std::fs::File::create(&outpath).unwrap();
            std::io::copy(&mut file, &mut outfile).unwrap();
        }

        // Get and Set permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if let Some(mode) = file.unix_mode() {
                std::fs::set_permissions(&outpath, std::fs::Permissions::from_mode(mode)).unwrap();
            }
        }
    }
}

#[proc_macro]
pub fn from_concrete_python_export_zip(input: TokenStream) -> TokenStream {
    let pt: Result<LitStr, _> = syn::parse(input);
    let Ok(path_litteral) = pt else {
        panic!("Unexpected input. Expected path string litteral.");
    };

    let path = PathBuf::from(path_litteral.value());
    if !path.is_relative() {
        panic!("Found absolute artifact path. Artifacts paths are resolved relative to the CARGO_MANIFEST_DIR directory.");
    }
    if std::env::var("CARGO_MANIFEST_DIR").is_err() {
        panic!("CARGO_MANIFEST_DIR environment variable not set (usually set by cargo). Something is wrong.");
    }
    let path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(path);
    if !path.exists() {
        panic!("Input path must point to an existing export zip (relative to CARGO_MANIFEST_DIR): File {} not found.", path.display());
    };

    let mut s = DefaultHasher::new();
    FastPathHasher::from_pathbuf(&path).hash(&mut s);
    let hash_val = s.finish();

    let output_folder = PathBuf::from(CONCRETE_BUILD_DIR).join(format!("{hash_val}"));
    if !output_folder.exists() {
        unzip(&path, &output_folder);
    }

    if !output_folder.join(PATH_CIRCUIT).exists() {
        panic!("Missing `circuit.mlir` file in the export. Did you save your server with the `via_mlir` option ?");
    }
    let mlir = read_to_string(output_folder.join(PATH_CIRCUIT)).unwrap();

    if !output_folder.join(PATH_SIMULATED).exists() {
        panic!("Missing `is_simulated` file in the export. Did you save your server with the `via_mlir` option ?");
    }
    let is_simulated = read_to_string(output_folder.join(PATH_SIMULATED))
        .unwrap()
        .parse::<u8>()
        .unwrap();

    if !output_folder.join(PATH_CONFIGURATION).exists() {
        panic!("Missing `configuration.json` file in the export. Did you save your server with the `via_mlir` option ?");
    }
    let configuration_string = read_to_string(output_folder.join("configuration.json")).unwrap();
    let conf: Configuration = serde_json::from_str(configuration_string.as_str()).unwrap();

    if !output_folder.join(PATH_COMPOSITION_RULES).exists() {
        panic!("Missing `composition_rules.json` file in the export. Did you save your server with the `via_mlir` option ?");
    }
    let composition_rules_string =
        read_to_string(output_folder.join(PATH_COMPOSITION_RULES)).unwrap();
    let composition_rules: Vec<serde_json::Value> =
        serde_json::from_str(composition_rules_string.as_str()).unwrap();

    let mut opts = compilation_options_new();
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
        configuration::ParameterSelectionStrategy::V0 => opts.pin_mut().set_optimizer_strategy(0),
        configuration::ParameterSelectionStrategy::Mono => opts.pin_mut().set_optimizer_strategy(1),
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

    let library = compile(
        mlir.as_str(),
        &opts,
        output_folder.as_os_str().to_str().unwrap(),
    );

    quote! {
        const MLIR: &str = #mlir;
        // const IS_SIMULATED: &str = #is_simulated;
        // const CONFIGURATION: &str = #config_string;
        // const COMPOSITION_RULES: &str = #compo_string;
    }
    .into()
}

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

    let unzipped_folder = PathBuf::from(CONCRETE_BUILD_DIR).join(format!("{hash_val}"));
    if !unzipped_folder.exists() {
        unzip(&path, &unzipped_folder);
    }

    if !unzipped_folder.join("circuit.mlir").exists() {
        panic!("Missing `circuit.mlir` file in the export. Did you save your server with the `via_mlir` option ?");
    }
    let mlir = read_to_string(unzipped_folder.join("circuit.mlir")).unwrap();

    if !unzipped_folder.join("is_simulated").exists() {
        panic!("Missing `is_simulated` file in the export. Did you save your server with the `via_mlir` option ?");
    }
    let is_simulated = read_to_string(unzipped_folder.join("is_simulated")).unwrap();

    if !unzipped_folder.join("configuration.json").exists() {
        panic!("Missing `configuration.json` file in the export. Did you save your server with the `via_mlir` option ?");
    }
    let configuration_string = read_to_string(unzipped_folder.join("configuration.json")).unwrap();
    let configuration: Configuration = serde_json::from_str(configuration_string.as_str()).unwrap();

    if !unzipped_folder.join("composition_rules.json").exists() {
        return quote!(compile_error!("Missing `composition_rules.json` file in the export. Did you save your server with the `via_mlir` option ?");).into();
    }
    let composition_rules = read_to_string(unzipped_folder.join("composition_rules.json")).unwrap();

    let config_string = format!("{:?}", configuration);

    quote! {
        const MLIR: &str = #mlir;
        const IS_SIMULATED: &str = #is_simulated;
        const CONFIGURATION: &str = #config_string;
        const COMPOSITION_RULES: &str = #composition_rules;
    }
    .into()
}

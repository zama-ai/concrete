use proc_macro::TokenStream;
use proc_macro::{self};
use quote::quote;
use std::fs::read_to_string;
use std::hash::Hash;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::time::SystemTime;
use syn::LitStr;
use zip;

let CONCRETE_BUILD_DIR: &'static str = env!("CONCRETE_SYS_BUILD_DIR");

struct FastPathHasher{
    path: PathBuf,
    ctime: SystemTime,
    mtime: SystemTime,
}

impl FastPathHasher {
    fn from_pathbuf(path: &PathBuf) -> FastPathHasher{
        let path = path.canonicalize().unwrap();
        let metadata = path.metadata().unwrap();
        FastPathHasher {
            ctime: metadata.ctime(),
            mtime: metadata.mtime()
            path
        }
    }
}

impl Hash for FastPathHasher {
    fn hash<H>(&self, state: &mut H) where H: Hasher{
        self.path.hash(state);
        self.ctime.hash(state);
        self.mtime.hash(state);
    }
}

fn unzip(zip_path: &Path, to: &Path){
        let file = fs::File::open(zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();
            let outpath = match file.enclosed_name() {
                Some(path) => to.join(path),
                None => continue,
            };
            if file.is_dir() {
                fs::create_dir_all(&outpath).unwrap();
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        fs::create_dir_all(p).unwrap();
                    }
                }
                let mut outfile = fs::File::create(&outpath).unwrap();
                io::copy(&mut file, &mut outfile).unwrap();
            }

            // Get and Set permissions
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                if let Some(mode) = file.unix_mode() {
                    fs::set_permissions(&outpath, fs::Permissions::from_mode(mode)).unwrap();
                }
            }
        }
}

#[proc_macro]
pub fn from_concrete_python_export_zip(input: TokenStream) -> TokenStream {
    let pt: Result<LitStr, _> = syn::parse(input);
    let Ok(path_litteral) = pt else {
        return quote!(compile_error!("Unexpected input. Expected path string litteral.");).into();
    };

    let path = PathBuf::from(path_litteral.value());
    if !path.exists() {
        return quote!(compile_error!("Input path must point to an existing export zip.");).into();
    }

    // We get a hash from path and mtime and ctime
    let mut s = DefaultHasher::new();
    FastPathHasher::from_pathbuf(path).hash(&mut s);
    let hash_val = s.finish();

    let unzipped_folder = path.join(hash_val);
    if !unzipped_folder.exists(){
        unzip(path, unzipped_folder);
    }

    if !unzipped_folder.join("circuit.mlir").exists(){
        return quote!(compile_error!("Missing `circuit.mlir` file in the export. Did you save your server with the `via_mlir` option ?");).into();
    }
    let mlir = read_to_string(unzipped_folder.join("circuit.mlir")).unwrap();

    if !unzipped_folder.join("is_simulated").exists(){
                return quote!(compile_error!("Missing `is_simulated` file in the export. Did you save your server with the `via_mlir` option ?");).into();
        }
    let is_simulated = read_to_string(unzipped_folder.join("is_simulated")).unwrap();

    if !unzipped_folder.join("configuration.json").exists(){
            return quote!(compile_error!("Missing `configuration.json` file in the export. Did you save your server with the `via_mlir` option ?");).into();
    }
    let configuration = read_to_string(unzipped_folder.join("configuration.json")).unwrap();

    if !unzipped_folder.join("composition_rules.json").exists(){
                return quote!(compile_error!("Missing `composition_rules.json` file in the export. Did you save your server with the `via_mlir` option ?");).into();
        }
    let composition_rules = read_to_string(unzipped_folder.join("composition_rules.json")).unwrap();

    quote! {}.into()
}

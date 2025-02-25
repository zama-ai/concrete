fn main() {
    ::capnpc::CompilerCommand::new()
        // the file is just a symlink to the actual schema: this is only to have
        // a simpler output file than using ../../x/x/concrete-protocol.capnp
        .file("capnp/concrete-protocol.capnp")
        .run()
        .expect("compiling schema");
}

import lit.formats

# Lit configuration
config.name = "concretelang"
config.test_format = lit.formats.ShTest("0")
config.suffixes = {".mlir"}
config.target_triple = ""

# Set the llvm
config.environment['PATH'] = os.pathsep.join([
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", "bin"),
    config.environment['PATH']]
)
print(config.environment['PATH'])

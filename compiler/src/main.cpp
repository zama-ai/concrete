#include <iostream>

#include <tclap/CmdLine.h>

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Parser.h>

#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

struct CommandLineArgs {
    std::vector<std::string> inputs;
    std::string output;
};

void parseCommandLine(int argc, char** argv, CommandLineArgs* args)
{
    try {
        TCLAP::CmdLine cmd("zamacompiler", ' ', "0.0.1");
        // Input file names
        TCLAP::UnlabeledMultiArg<std::string> fileNames("file", "The input files", false, "file");
        cmd.add(fileNames);

        // Output
        TCLAP::ValueArg<std::string> output("o","out","Place the output into the <file>",false, "","string");
        cmd.add(output);

        cmd.parse( argc, argv );
        args->output = output.getValue();
        args->inputs = fileNames.getValue();

    } catch (TCLAP::ArgException &e)  // catch exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    // Parse command line arguments
    CommandLineArgs cmdLineArgs;
    parseCommandLine(argc, argv, &cmdLineArgs);

    // Initialize the MLIR context
    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
    context.getOrLoadDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();

    // For all input file, parse and dump
    for (const auto& fileName: cmdLineArgs.inputs) {
        auto module = mlir::parseSourceFile<mlir::ModuleOp>(fileName, &context);
        module->dump();
    }
    return 0;
}

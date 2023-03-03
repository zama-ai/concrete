import os
import re
import sys
import numpy
import random

class Param:
    def __init__(self):
        self.typ = ""
        self.values = []
        self.shapes = []
        self.width = 0

def generateHeader(output, name):
    output.write("description: " + name + "\n")
    output.write("program: |\n")

def generateFooter(output, params):
    output.write("tests:\n")
    output.write("  - inputs:\n")
    for p in params:
        if p.typ == "scalar":
            output.write("    - scalar: " + p.value + "\n")
        if p.typ == "tensor":
            output.write("    - tensor: [")
            for i, v in enumerate(p.values):
                sv = str(v)
                if i == 0:
                    output.write(sv)
                else:
                    output.write(", " + sv)
            output.write("]\n")
            output.write("      shape: [")
            for i, v in enumerate(p.shapes):
                sv = str(v)
                if i == 0:
                    output.write(sv)
                else:
                    output.write(", " + sv)
            output.write("]\n")
            #output.write("      width: " + str(p.width+1) + "\n")
    output.write("---\n\n")

def getParams(filename):
    f = open(filename, 'r')
    params = []
    for line in f:
        m = re.match(r".*?func.func @main\((.*?)\).*?", line)
        if m:
            args = re.split(r'%\w+:', m.group(1))
            for a in args:
                am = re.match(r"\W*tensor<((?:\d+x)+)(?:(?:!FHE.eint<(\d+)>>)|(?:i(\d+)>))", a)
                if am:
                    param = Param()
                    param.typ = "tensor"
                    shapes = list(filter(None, re.split(r'x', am.group(1))))
                    param.shapes = list(map(int, shapes))
                    if am.group(2):
                        param.width = int(am.group(2))
                    else:
                        param.width = int(am.group(3))
                    for i in range(0, numpy.prod(param.shapes)):
                        param.values.append(random.randint(0, 2**param.width))
                    params.append(param)
    return params

if __name__ == "__main__":
    # Find all MLIRs
    for dirname, dirnames, filenames in os.walk(sys.argv[1]):
        for i, filename in enumerate(filenames):
            if i % 20 == 0:
                output = open(sys.argv[2] + "_" + str(int(i/20)) + ".yaml", 'w')
            desc = re.match(r"(.*?)\.mlir$", filename)
            if desc:
                generateHeader(output, desc.group(1))
                f = open(os.path.join(dirname, filename), 'r')
                output.write(f.read() + "\n")
                f.close()
                generateFooter(output, getParams(os.path.join(dirname, filename)))
            if i % 20 == 19:
                output.close()

        if '.git' in dirnames:
            dirnames.remove('.git')


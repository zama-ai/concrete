#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

for i in range(128):
    args = ','.join(f'args[{j}]' for j in range(i))
    print(f'        case {i}: return func({args});')

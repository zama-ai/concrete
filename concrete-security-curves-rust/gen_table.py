import sys, json;

def print_curve(data):
    print(f'    ({data["security_level"]}, SecurityWeights {{ slope: {data["slope"]}, bias: {data["bias"]}, minimal_lwe_dimension: {data["minimal_lwe_dimension"]} }}),')


def print_rust_curves_declaration(datas):
    print(f"const SECURITY_WEIGHTS_ARRAY: [(u64, SecurityWeights); {len(datas)}] = [")
    for data in datas:
        print_curve(data)
    print("];")

print_rust_curves_declaration(json.load(sys.stdin))
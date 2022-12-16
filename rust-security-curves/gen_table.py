import sys, json;

def print_curve(data):
    print(f'    ({data["bits"]}, SecurityWeights {{ slope: {data["linear_term1"]}, bias: {data["linear_term2"]}, minimal_lwe_dimension: {data["n_alpha"]} }}),')


def print_rust_curves_declaration(datas):
    print("[")
    for data in datas:
        print_curve(data)
    print("]")

print_rust_curves_declaration(json.load(open("json/curves.json")))
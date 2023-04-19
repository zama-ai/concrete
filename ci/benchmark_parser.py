"""
benchmark_parser
----------------

Parse benchmark raw results.
"""
import argparse
import pathlib
import json
import sys

ONE_HOUR_IN_NANOSECONDS = 3600E9

parser = argparse.ArgumentParser()
parser.add_argument('results_path',
                    help=('Location of raw benchmark results,'
                          ' could be either a file or a directory.'
                          'In a case of a directory, this script will attempt to parse all the'
                          'files containing a .json extension'))
parser.add_argument('output_file', help='File storing parsed results')
parser.add_argument('-d', '--database', dest='database', required=True,
                    help='Name of the database used to store results')
parser.add_argument('-w', '--hardware', dest='hardware', required=True,
                    help='Hardware reference used to perform benchmark')
parser.add_argument('-V', '--project-version', dest='project_version', required=True,
                    help='Commit hash reference')
parser.add_argument('-b', '--branch', dest='branch', required=True,
                    help='Git branch name on which benchmark was performed')
parser.add_argument('--commit-date', dest='commit_date', required=True,
                    help='Timestamp of commit hash used in project_version')
parser.add_argument('--bench-date', dest='bench_date', required=True,
                    help='Timestamp when benchmark was run')
parser.add_argument('--throughput', dest='throughput', action='store_true',
                    help='Compute and append number of operations per millisecond and'
                         'operations per dollar, only on mean values')


def parse_results(raw_results, compute_throughput=False, hardware_hourly_cost=None):
    """
    Parse raw benchmark results.

    :param raw_results: path to file that contains raw results as :class:`pathlib.Path`
    :param compute_throughput: compute number of operations per millisecond and operations per
        dollar on mean values
    :param hardware_hourly_cost: hourly cost of the hardware used in dollar

    :return: :class:`list` of data points
    """
    raw_results = json.loads(raw_results.read_text())
    parsed_results = []
    for res in raw_results["benchmarks"]:
        test_name = res["name"]
        value = res["cpu_time"]
        parsed_results.append({"value": value, "test": test_name})        
                                   
        try:        
            value = res["Throughput"]
            parsed_results.append({"value": value, "test": "_".join([test_name, "throughput"])})
        except KeyError:
            pass

        if test_name.endswith("_mean") and compute_throughput:
            parsed_results.append({
                "value": compute_ops_per_millisecond(value),
                "test": "_".join([test_name, "ops_per_ms"])})

            if hardware_hourly_cost is not None:
                parsed_results.append({
                    "value": compute_ops_per_dollar(value, hardware_hourly_cost),
                    "test": "_".join([test_name, "ops_per_dollar"])})

    return parsed_results


def recursive_parse(directory, compute_throughput=False, hardware_hourly_cost=None):
    """
    Parse all the benchmark results in a directory. It will attempt to parse all the files having a
    .json extension at the top-level of this directory.

    :param directory: path to directory that contains raw results as :class:`pathlib.Path`
    :param compute_throughput: compute number of operations per millisecond and operations per
        dollar
    :param hardware_hourly_cost: hourly cost of the hardware used in dollar

    :return: :class:`list` of data points
    """
    result_values = []
    for file in directory.glob('*.json'):
        try:
            result_values.extend(parse_results(file, compute_throughput, hardware_hourly_cost))
        except KeyError as err:
            print(f"Failed to parse '{file.resolve()}': {repr(err)}")

    return result_values


def dump_results(parsed_results, filename, input_args):
    """
    Dump parsed results formatted as JSON to file.

    :param parsed_results: :class:`list` of data points
    :param filename: filename for dump file as :class:`pathlib.Path`
    :param input_args: CLI input arguments
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    series = {
        "database": input_args.database,
        "hardware": input_args.hardware,
        "project_version": input_args.project_version,
        "branch": input_args.branch,
        "insert_date": input_args.bench_date,
        "commit_date": input_args.commit_date,
        "points": parsed_results,
    }
    filename.write_text(json.dumps(series))


def compute_ops_per_dollar(data_point, product_hourly_cost):
    """
    Compute numbers of operations per dollar for a given ``data_point``.

    :param data_point: timing value measured during benchmark in nanoseconds
    :param product_hourly_cost: cost in dollar per hour of hardware used

    :return: number of operations per dollar
    """
    return ONE_HOUR_IN_NANOSECONDS / (product_hourly_cost * data_point)


def compute_ops_per_millisecond(data_point):
    """
    Compute numbers of operations per millisecond for a given ``data_point``.

    :param data_point: timing value measured during benchmark in nanoseconds

    :return: number of operations per millisecond
    """
    return 1E6 / data_point


if __name__ == "__main__":
    args = parser.parse_args()

    hardware_cost = None
    if args.throughput:
        print("Throughput computation enabled")
        ec2_costs = json.loads(
            pathlib.Path("ci/ec2_products_cost.json").read_text(encoding="utf-8"))
        try:
            hardware_cost = abs(ec2_costs[args.hardware])
            print(f"Hardware hourly cost: {hardware_cost} $/h")
        except KeyError:
            print(f"Cannot find hardware hourly cost for '{args.hardware}'")
            sys.exit(1)

    results_path = pathlib.Path(args.results_path)
    print("Parsing benchmark results... ")
    if results_path.is_dir():
        results = recursive_parse(results_path, args.throughput, hardware_cost)
    else:
        results = parse_results(results_path, args.throughput, hardware_cost)
    print("Parsing results done")

    output_file = pathlib.Path(args.output_file)
    print(f"Dump parsed results into '{output_file.resolve()}' ... ", end="")
    dump_results(results, output_file, args)

    print("Done")

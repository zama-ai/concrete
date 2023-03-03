"""
benchmark_parser
----------------

Parse benchmark raw results.
"""
import argparse
import pathlib
import json


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


def parse_results(raw_results):
    """
    Parse raw benchmark results.

    :param raw_results: path to file that contains raw results as :class:`pathlib.Path`

    :return: :class:`list` of data points
    """
    raw_results = json.loads(raw_results.read_text())
    return [
        {"value": res["cpu_time"], "test": res["name"]}
        for res in raw_results["benchmarks"]
    ]


def recursive_parse(directory):
    """
    Parse all the benchmark results in a directory. It will attempt to parse all the files having a
    .json extension at the top-level of this directory.

    :param directory: path to directory that contains raw results as :class:`pathlib.Path`

    :return: :class:`list` of data points
    """
    result_values = []
    for file in directory.glob('*.json'):
        try:
            result_values.extend(parse_results(file))
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


if __name__ == "__main__":
    args = parser.parse_args()

    results_path = pathlib.Path(args.results_path)
    print("Parsing benchmark results... ")
    if results_path.is_dir():
        results = recursive_parse(results_path)
    else:
        results = parse_results(results_path)
    print("Parsing results done")

    output_file = pathlib.Path(args.output_file)
    print(f"Dump parsed results into '{output_file.resolve()}' ... ", end="")
    dump_results(results, output_file, args)

    print("Done")

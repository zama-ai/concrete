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
parser.add_argument('-n', '--series-name', dest='series_name',
                    default="concrete_compiler_benchmark_timing",
                    help='Name of the data series (as stored in Prometheus)')
parser.add_argument('-e', '--series-help', dest='series_help',
                    default="Timings of various type of benchmarks in concrete compiler.",
                    help='Description of the data series (as stored in Prometheus)')
parser.add_argument('-t', '--series-tags', dest='series_tags',
                    type=json.loads, default={},
                    help='Tags to apply to all the points in the data series')


def parse_results(raw_results):
    """
    Parse raw benchmark results.

    :param raw_results: path to file that contains raw results as :class:`pathlib.Path`

    :return: :class:`list` of data points
    """
    result_values = list()
    raw_results = json.loads(raw_results.read_text())
    for res in raw_results["benchmarks"]:
        bench_class, action, option_class, application = res["run_name"].split("/")

        for measurement in ("real_time", "cpu_time"):
            tags = {"bench_class": bench_class,
                    "action": action,
                    "option_class": option_class,
                    "application": application,
                    "measurement": measurement}
            result_values.append({"value": res[measurement], "tags": tags})

    return result_values


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


def dump_results(parsed_results, filename, series_name,
                 series_help="", series_tags=None):
    """
    Dump parsed results formatted as JSON to file.

    :param parsed_results: :class:`list` of data points
    :param filename: filename for dump file as :class:`pathlib.Path`
    :param series_name: name of the data series as :class:`str`
    :param series_help: description of the data series as :class:`str`
    :param series_tags: constant tags for the series
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    series = [
        {"series_name": series_name,
         "series_help": series_help,
         "series_tags": series_tags or dict(),
         "points": parsed_results},
    ]
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
    dump_results(results, output_file, args.series_name,
                 series_help=args.series_help, series_tags=args.series_tags)

    print("Done")

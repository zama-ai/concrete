import argparse
import json
import os
import pathlib
import subprocess
import urllib
import tqdm


def name_to_id(name):
    """Convert a human readable name to a url friendly id (e.g., `x + y` to `x-plus-y`)"""

    name = name.replace("-", "minus")
    name = name.replace("**", "-to-the-power-of-")
    name = name.replace("+", "plus")
    name = name.replace("*", "times")
    name = name.replace("/", "over")
    name = name.replace("%", "percent")
    name = name.replace(" ", "-")
    name = name.replace("(", "")
    name = name.replace(")", "")

    return urllib.parse.quote_plus(name.lower())


def identify_metrics(script, lines, metrics):
    """Identify the metrics of a script and make sure the annotations are well-formed"""

    # Create a flag to detect `# Measure: End` without a measurement start
    in_measurement = False

    # Create a variable to remember the indentation of the start of the last measurement
    measurement_indentation = 0

    # Create a variable to remember the line number of the start of the last measurement
    measurement_line = 0

    # Identify measurements and store their name and id in `metrics`
    for index, line in enumerate(lines):
        # Get the indentation of the line
        indentation = len(line) - len(line.lstrip())

        # Strip the line for easier processing
        line = line.strip()

        # Check whether the line is a special line or not
        if line == "# Measure: End":
            # Make sure a measurement is active already
            if not in_measurement:
                raise SyntaxError(
                    f"Measurements cannot end before they are defined "
                    f"(at line {index + 1} of {script})",
                )

            # Make sure indentation of the current line
            # matches the indentation of the active measurement line
            if indentation != measurement_indentation:
                raise SyntaxError(
                    f"Measurements should finish with the same indentation as they are defined "
                    f"(at lines {measurement_line} and {index + 1} of {script})",
                )

            # Set in_measurement to false as the active measurement has ended
            in_measurement = False
        elif line.startswith("# Measure:"):
            # Make sure a measurement is not active already
            if in_measurement:
                raise SyntaxError(
                    f"Nested measurements are not supported "
                    f"(at lines {measurement_line} and {index + 1} of {script})",
                )

            # Extract the measurement details
            measurement_details = line.replace("# Measure:", "").split("=")

            # Extract metric name and id
            metric_label = measurement_details[0].strip()
            metric_id = name_to_id(metric_label)

            # Add metric id and metric name to `metrics`
            metrics[metric_id] = metric_label

            # Check if the measurement is a timing measurement (does not contain `= expression`)
            if len(measurement_details) == 1:
                # We need to see an end in the upcoming lines so update variables accordingly
                in_measurement = True
                measurement_line = index + 1
                measurement_indentation = indentation

    # Make sure there isn't an active measurement that hasn't finished
    if in_measurement:
        raise SyntaxError(
            f"Unfinished measurements are not supported "
            f"(at line {measurement_line} of {script})",
        )


def create_modified_script(script_without_extension, lines, metrics):
    """Create a modified version of the script which can be used to perform measurements"""

    with open(f"{script_without_extension}.measure.py", "w") as f:
        # Import must-have libraries
        f.write("import json\n")
        f.write("import time\n")
        f.write("\n")

        # Create a measurement dictionary to accumulate values
        f.write("_measurements_ = {\n")
        for metric_id in metrics.keys():
            f.write(f'    "{metric_id}": [],\n')
        f.write("}\n")

        # Create a variable to hold the id of the current metric
        # This is required to determine where to save the measured value
        current_metric_id = ""

        # Copy the lines of the original script into the new script
        for line in lines[1:]:
            # And modify special lines along the way
            if line.strip() == "# Measure: End":
                # Replace `# Measure: End` with
                #
                #   _end_ = time.time()
                #   _measurements_["id"].append((_end_ - _start_) * 1000)

                index = line.find("# Measure: End")
                line = line[:index]

                f.write(f"{line}_end_ = time.time()\n")

                value = "(_end_ - _start_) * 1000"
                line += f'_measurements_["{current_metric_id}"].append({value})\n'
            elif line.strip().startswith("# Measure:"):
                # Replace `# Measure: ...` with
                #
                #   _start_ = time.time()

                # Replace `# Measure: ... = expression` with
                #
                #  _measurements_["id"].append(expression)

                metric_details = line.replace("# Measure:", "").split("=")
                metric_label = metric_details[0].strip()
                metric_id = name_to_id(metric_label)

                index = line.find("# Measure:")
                line = line[:index]

                if len(metric_details) == 1:
                    current_metric_id = metric_id
                    line += "_start_ = time.time()\n"
                else:
                    value = metric_details[1]
                    line += f'_measurements_["{metric_id}"].append({value.strip()})\n'

            # Write the possibly replaced line back
            f.write(line)

        # Dump measurements to a temporary file after the script is executed from start to end
        f.write("\n")
        f.write(f'with open("{script_without_extension}.measurements", "w") as f:\n')
        f.write("    json.dump(_measurements_, f, indent=2)\n")


def perform_measurements(script, script_without_extension, target_id, metrics, samples, result):
    """Run the modified script multiple times and update the result"""

    # Create a flag to keep track of the working status
    working = True

    print()
    print(script)
    print("-" * len(str(script)))

    # Run the modified script `samples` times and accumulate measurements
    measurements = {metric_id: [] for metric_id in metrics.keys()}
    with tqdm.tqdm(total=samples) as pbar:
        for i in range(samples):
            # Create the subprocess
            process = subprocess.run(
                ["python", f"{script_without_extension}.measure.py"],
                capture_output=True,
            )

            # Print sample information
            pbar.write(f"    Sample {i + 1}")
            pbar.write(f"    {'-' * len(f'Sample {i + 1}')}")

            # If the script raised an exception, discard everything for now
            if process.returncode != 0:
                working = False

                pbar.write(f"        Failed (exited with {process.returncode})")
                pbar.write(f"        --------------------{'-' * len(str(process.returncode))}-")

                stderr = process.stderr.decode("utf-8")
                for line in stderr.split("\n"):
                    if line.strip() != "":
                        pbar.write(f"            {line}")

                pbar.update(samples)
                break

            # Read the measurements and delete the temporary file
            with open(f"{script_without_extension}.measurements") as f:
                results = json.load(f)
            os.unlink(f"{script_without_extension}.measurements")

            # Add the `results` of the current run to `measurements`
            for metric_id in metrics.keys():
                average = sum(results[metric_id]) / len(results[metric_id])
                pbar.write(f"        {metrics[metric_id]} = {average}")

                for measurement in results[metric_id]:
                    measurements[metric_id].append(measurement)
            pbar.write("")

            pbar.update(1)
    print()

    result["targets"][target_id]["working"] = working

    if working:
        # Take average of all metrics and store them in `result`
        result["targets"][target_id]["measurements"].update(
            {metric_id: sum(metric) / len(metric) for metric_id, metric in measurements.items()}
        )

        # Add metrics of the current script to the result
        for metric_id, metric_label in metrics.items():
            if metric_id not in result["metrics"]:
                result["metrics"][metric_id] = {"label": metric_label}
    else:
        # Delete measurements field of the current target
        del result["targets"][target_id]["measurements"]


def main():
    parser = argparse.ArgumentParser(description="Measurement script for the progress tracker")

    parser.add_argument("base", type=str, help="directory which contains the benchmarks")
    parser.add_argument("--samples", type=int, default=30, help="number of samples to take")
    parser.add_argument("--keep", action="store_true", help="flag to keep measurement scripts")

    args = parser.parse_args()

    base = pathlib.Path(args.base)
    samples = args.samples

    with open(".benchmarks/machine.json", "r") as f:
        machine = json.load(f)

    result = {"machine": machine, "metrics": {}, "targets": {}}
    scripts = list(base.glob("*.py"))

    # Process each script under the base directory
    for script in filter(lambda script: not str(scripts[0]).endswith("measure.py"), scripts):
        # Read the script line by line
        with open(script, "r") as f:
            lines = f.readlines()

        # Find the first non-empty line
        first_line = ""
        for line in map(lambda line: line.strip(), lines):
            if line != "":
                first_line = line
                break

        # Check whether the script is a target or not
        if not first_line.startswith("# Target:"):
            print()
            print(script)
            print("-" * len(str(script)))

            with tqdm.tqdm(total=samples) as pbar:
                pbar.write("    Sample 1")
                pbar.write("    --------")
                pbar.write("        Skipped (doesn't have a `# Target:` directive)\n")
                pbar.update(samples)

            print()
            continue

        # Extract target name and id
        target_name = first_line.replace("# Target:", "").strip()
        target_id = name_to_id(target_name)

        # Check whether the target is already registered
        if target_id in result["targets"]:
            raise RuntimeError(f"Target `{target_name}` is already registered")

        # Create an entry in the result for the current target
        result["targets"][target_id] = {"name": target_name, "measurements": {}}

        # Create a dictionary to hold `metric_id` to `metric_name`
        metrics = {}

        # Identify metrics of the current script
        identify_metrics(script, lines, metrics)

        # Extract the script name without extension
        script_without_extension = os.path.splitext(script)[0]

        # Create another script to hold the modified version of the current script
        create_modified_script(script_without_extension, lines, metrics)

        # Perform and save measurements
        perform_measurements(script, script_without_extension, target_id, metrics, samples, result)

        # Dump the latest results to the output file
        with open(".benchmarks/findings.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Delete the modified script if the user doesn't care
        if not args.keep:
            os.unlink(f"{os.path.splitext(script)[0]}.measure.py")
    print()


if __name__ == "__main__":
    main()

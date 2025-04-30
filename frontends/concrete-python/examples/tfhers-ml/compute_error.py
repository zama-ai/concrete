import json

import click


@click.command()
@click.option(
    "--plaintext-file", "-p", required=True, help="Path to the rescaled plaintext values file."
)
@click.option(
    "--quantized-predictions-file",
    "-q",
    required=True,
    help="Path to the test_values.json file containing quantized predictions.",
)
def compute_error(plaintext_file, quantized_predictions_file):
    """Compute the error between decrypted rescaled values and quantized predictions."""
    # Read rescaled plaintext values from plaintext_file
    with open(plaintext_file, encoding="utf-8") as f:
        rescaled_plaintext_values = [int(x) for x in f.read().strip().split(",")]

    # Read quantized_predictions from quantized_predictions_file
    with open(quantized_predictions_file, encoding="utf-8") as f:
        data = json.load(f)
        quantized_predictions = data["quantized_predictions"]

    # Flatten quantized_predictions
    quantized_predictions_flat = [int(x) for sublist in quantized_predictions for x in sublist]

    # Round down 10 bits using (x // (1 << 10)) * (1 << 10)
    rounded_quantized_predictions = [
        (x // (1 << 10)) * (1 << 10) for x in quantized_predictions_flat
    ]

    # Compare rescaled_plaintext_values with rounded_quantized_predictions
    num_differences = 0
    total_values = len(rescaled_plaintext_values)
    errors = []
    for i in range(total_values):
        a = rescaled_plaintext_values[i]
        b = rounded_quantized_predictions[i]
        print(f"output: {a}, expected: {b}")
        if a != b:
            num_differences += 1
            error_in_units = round((a - b) / (1 << 10))
            errors.append((i, error_in_units))

    print(f"Number of differing values: {num_differences}")
    print(f"Total values compared: {total_values}")
    if num_differences > 0:
        print("Differences (index, error in units of 2^10):")
        for idx, error_in_units in errors:
            print(f"Index {idx}: error = {error_in_units}")

    # success is when we don't offset by more than 1
    for error in errors:
        if error[1] > 1:
            return 1
    return 0


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    compute_error()
    # pylint: enable=no-value-for-parameter

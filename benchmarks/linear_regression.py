from copy import deepcopy
from typing import Any, Dict

import numpy as np
import progress
from common import BENCHMARK_CONFIGURATION
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from concrete.quantization import QuantizedArray, QuantizedLinear, QuantizedModule


class QuantizedLinearRegression(QuantizedModule):
    """
    Quantized Generalized Linear Model
    Building on top of QuantizedModule, implement a quantized linear transformation (w.x + b)
    """

    @staticmethod
    def from_sklearn(sklearn_model, calibration_data):
        """Create a Quantized Linear Regression initialized from a sklearn trained model"""
        weights = np.expand_dims(sklearn_model.coef_, 1)
        bias = sklearn_model.intercept_
        # Quantize with 6 bits for input data, 1 for weights, 1 for the bias and 6 for the output
        return QuantizedLinearRegression(6, 1, 1, 6, weights, bias, calibration_data)

    def __init__(self, q_bits, w_bits, b_bits, out_bits, weights, bias, calibration_data) -> None:
        """
        Create the linear regression with different quantization bit precisions:

        Quantization Parameters - Number of bits:
                    q_bits (int): bits for input data, insuring that the number of bits of
                                the w . x + b operation does not exceed 7 for the calibration data
                    w_bits (int): bits for weights: in the case of a univariate regression this
                                can be 1
                    b_bits (int): bits for bias (this is a single value so a single bit is enough)
                    out_bits (int): bits for the result of the linear transformation (w.x + b).
                                  In our case since the result of the linear transformation is
                                  directly decrypted we can use the maximum of 7 bits

        Other parameters:
                    weights: a numpy nd-array of weights (Nxd) where d is the data dimensionality
                    bias: a numpy scalar
                    calibration_data: a numpy nd-array of data (Nxd)
        """
        self.n_bits = out_bits

        # We need to calibrate to a sufficiently low number of bits
        # so that the output of the Linear layer (w . x + b)
        # does not exceed 7 bits
        self.q_calibration_data = QuantizedArray(q_bits, calibration_data)

        # Quantize the weights and create the quantized linear layer
        q_weights = QuantizedArray(w_bits, weights)
        q_bias = QuantizedArray(b_bits, bias)
        q_layer = QuantizedLinear(out_bits, q_weights, q_bias)

        # Store quantized layers
        quant_layers_dict: Dict[str, Any] = {}

        # Calibrate the linear layer and obtain calibration_data for the next layers
        calibration_data = self._calibrate_and_store_layers_activation(
            "linear", q_layer, calibration_data, quant_layers_dict
        )

        # Finally construct our Module using the quantized layers
        super().__init__(quant_layers_dict)

    def _calibrate_and_store_layers_activation(
        self, name, q_function, calibration_data, quant_layers_dict
    ):
        """
        This function calibrates a layer of a quantized module (e.g. linear, inverse-link,
        activation, etc) by looking at the input data, then computes the output of the quantized
        version of the layer to be used as input to the following layers
        """

        # Calibrate the output of the layer
        q_function.calibrate(calibration_data)
        # Store the learned quantized layer
        quant_layers_dict[name] = q_function
        # Create new calibration data (output of the previous layer)
        q_calibration_data = QuantizedArray(self.n_bits, calibration_data)
        # Dequantize to have the value in clear and ready for next calibration
        return q_function(q_calibration_data).dequant()

    def quantize_input(self, x):
        """Quantize an input set with the quantization parameters determined from calibration"""
        q_input_arr = deepcopy(self.q_calibration_data)
        q_input_arr.update_values(x)
        return q_input_arr


@progress.track([{"id": "linear-regression", "name": "Linear Regression", "parameters": {}}])
def main():
    """
    Our linear regression benchmark. Use some synthetic data to train a regression model,
    then fit a model with sklearn. We quantize the sklearn model and compile it to FHE.
    We compute the training loss for the quantized and FHE models and compare them. We also
    predict on a test set and compare FHE results to predictions from the quantized model
    """

    X, y, _ = make_regression(
        n_samples=200, n_features=1, n_targets=1, bias=5.0, noise=30.0, random_state=42, coef=True
    )

    # Split it into train/test and sort the sets for nicer visualization
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    sidx = np.argsort(np.squeeze(x_train))
    x_train = x_train[sidx, :]
    y_train = y_train[sidx]

    sidx = np.argsort(np.squeeze(x_test))
    x_test = x_test[sidx, :]
    y_test = y_test[sidx]

    # Train a linear regression with sklearn and predict on the test data
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)

    # Calibrate the model for quantization using both training and test data
    calib_data = X  # np.vstack((x_train, x_test))
    q_linreg = QuantizedLinearRegression.from_sklearn(linreg, calib_data)

    # Compile the quantized model to FHE
    engine = q_linreg.compile(
        q_linreg.quantize_input(calib_data),
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )

    # Measure test error using the clear-sklearn, the clear-quantized and the FHE quantized model
    # as R^2 coefficient for the test data

    # First, predict using the sklearn classifier
    y_pred = linreg.predict(x_test)

    # Now that the model is quantized, predict on the test set
    x_test_q = q_linreg.quantize_input(x_test)
    q_y_pred = q_linreg.forward_and_dequant(x_test_q)

    # Now predict using the FHE quantized model on the testing set
    y_test_pred_fhe = np.zeros_like(x_test)

    for i, x_i in enumerate(tqdm(x_test_q.qvalues)):
        q_sample = np.expand_dims(x_i, 1).transpose([1, 0]).astype(np.uint8)
        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            q_pred_fhe = engine.run(q_sample)
        y_test_pred_fhe[i] = q_linreg.dequantize_output(q_pred_fhe)

    # Measure the error for the three versions of the classifier
    sklearn_r2 = r2_score(y_pred, y_test)
    non_homomorphic_test_error = r2_score(q_y_pred, y_test)
    homomorphic_test_error = r2_score(y_test_pred_fhe, y_test)

    # Measure the error of the FHE quantized model w.r.t the clear quantized model
    difference = (
        abs(homomorphic_test_error - non_homomorphic_test_error) * 100 / non_homomorphic_test_error
    )

    print(f"Sklearn R^2: {sklearn_r2:.4f}")
    progress.measure(
        id="sklearn-r2",
        label="Sklearn R^2",
        value=sklearn_r2,
    )

    print(f"Non Homomorphic R^2: {non_homomorphic_test_error:.4f}")
    progress.measure(
        id="non-homomorphic-r2",
        label="Non Homomorphic R^2",
        value=non_homomorphic_test_error,
    )

    print(f"Homomorphic R^2: {homomorphic_test_error:.4f}")
    progress.measure(
        id="homomorphic-r2",
        label="Homomorphic R^2",
        value=homomorphic_test_error,
    )

    print(f"Relative Loss Difference (%): {difference:.2f}%")
    progress.measure(
        id="relative-loss-difference-percent",
        label="Relative Loss Difference (%)",
        value=difference,
        alert=(">", 7.5),
    )

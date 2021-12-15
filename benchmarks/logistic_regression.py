# bench: Full Target: Logistic Regression

# Disable line length warnings as we have a looooong metric...
# flake8: noqa: E501
# pylint: disable=C0301

from copy import deepcopy
from typing import Any, Dict

import numpy as np
from numpy.random import RandomState
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from concrete.quantization import QuantizedArray, QuantizedLinear, QuantizedModule, QuantizedSigmoid


class QuantizedLogisticRegression(QuantizedModule):
    """
    Quantized Logistic Regression
    Building on top of QuantizedModule, this class will chain together a linear transformation
    and an inverse-link function, in this case the logistic function
    """

    @staticmethod
    def from_sklearn(sklearn_model, calibration_data):
        """Create a Quantized Logistic Regression initialized from a sklearn trained model"""
        if sklearn_model.coef_.ndim == 1:
            weights = np.expand_dims(sklearn_model.coef_, 1)
        else:
            weights = sklearn_model.coef_.transpose()

        bias = sklearn_model.intercept_

        # In our case we have two data dimensions, we the weights precision needs to be 2 bits, as
        # for now we need the quantized values to be greater than zero for weights
        # Thus, to insure a maximum of 7 bits in the output of the linear transformation, we choose
        # 4 bits for the data and the minimum of 1 for the bias
        return QuantizedLogisticRegression(4, 2, 1, 6, weights, bias, calibration_data)

    def __init__(self, q_bits, w_bits, b_bits, out_bits, weights, bias, calibration_data) -> None:
        """
        Create the Logistic regression with different quantization bit precisions:

        Quantization Parameters - Number of bits:
                    q_bits (int): bits for input data, insuring that the number of bits of
                                the w . x + b operation does not exceed 7 for the calibration data
                    w_bits (int): bits for weights: in the case of a univariate regression this
                                can be 1
                    b_bits (int): bits for bias (this is a single value so a single bit is enough)
                    out_bits (int): bits for the result of the linear transformation (w.x + b).
                                  In the case of Logistic Regression the result of the linear
                                  transformation is input to a univariate inverse-link function, so
                                  this value can be 7

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

        # Add the inverse-link for inference.
        # This needs to be quantized since it's computed in FHE,
        # but we can use 7 bits of output since, in this case,
        # the result of the inverse-link is not processed by any further layers
        # Seven bits is the maximum precision but this could be lowered to improve speed
        # at the possible expense of higher deviance of the regressor
        q_logit = QuantizedSigmoid(n_bits=7)

        # Now calibrate the inverse-link function with the linear layer's output data
        calibration_data = self._calibrate_and_store_layers_activation(
            "invlink", q_logit, calibration_data, quant_layers_dict
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
        q_input_arr = deepcopy(self.q_calibration_data)
        q_input_arr.update_values(x)
        return q_input_arr


def main():
    """Main benchmark function: generate some synthetic data for two class classification,
    split train-test, train a sklearn classifier, calibrate and quantize it on the whole dataset
    then compile it to FHE. Test the three versions of the classifier on the test set and
    report accuracy"""

    # Generate some data with a fixed seed
    X, y = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=2,
        n_clusters_per_class=1,
        n_samples=100,
    )

    # Scale the data randomly, fixing seeds for reproductibility
    rng = RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    # Split it into train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Train a logistic regression with sklearn on the training set
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    # Calibrate the model for quantization using both training and test data
    calib_data = X
    q_logreg = QuantizedLogisticRegression.from_sklearn(logreg, calib_data)

    # Now, we can compile our model to FHE, taking as possible input set all of our dataset
    X_q = q_logreg.quantize_input(X)

    # bench: Measure: Compilation Time (ms)
    engine = q_logreg.compile(X_q)
    # bench: Measure: End

    # Start classifier evaluation

    # Test the original classifier
    y_pred_test = np.asarray(logreg.predict(x_test))

    # Now that the model is quantized, predict on the test set
    x_test_q = q_logreg.quantize_input(x_test)
    q_y_score_test = q_logreg.forward_and_dequant(x_test_q)
    q_y_pred_test = (q_y_score_test > 0.5).astype(np.int32)

    non_homomorphic_correct = 0
    homomorphic_correct = 0

    # Track the samples that are wrongly classified due to quantization issues
    q_wrong_predictions = np.zeros((0, 2), dtype=X.dtype)

    # Predict the FHE quantized classifier probabilities on the test set.
    # Compute FHE quantized accuracy, clear-quantized accuracy and
    # keep track of samples wrongly classified due to quantization
    for i, x_i in enumerate(tqdm(x_test_q.qvalues)):
        y_i = y_test[i]

        fhe_in_sample = np.expand_dims(x_i, 1).transpose([1, 0]).astype(np.uint8)

        # bench: Measure: Evaluation Time (ms)
        q_pred_fhe = engine.run(fhe_in_sample)
        # bench: Measure: End
        y_score_fhe = q_logreg.dequantize_output(q_pred_fhe)
        homomorphic_prediction = (y_score_fhe > 0.5).astype(np.int32)

        non_homomorphic_prediction = q_y_pred_test[i]
        if non_homomorphic_prediction == y_i:
            non_homomorphic_correct += 1
        elif y_pred_test[i] == y_i:
            # If this was a correct prediction with the clear-sklearn classifier
            q_wrong_predictions = np.vstack((q_wrong_predictions, x_test[i, :]))

        if homomorphic_prediction == y_i:
            homomorphic_correct += 1

    # Aggregate accuracies for all the versions of the classifier
    sklearn_acc = np.sum(y_pred_test == y_test) / len(y_test) * 100
    non_homomorphic_accuracy = (non_homomorphic_correct / len(y_test)) * 100
    homomorphic_accuracy = (homomorphic_correct / len(y_test)) * 100
    difference = abs(homomorphic_accuracy - non_homomorphic_accuracy)

    print()
    print(f"Sklearn accuracy: {sklearn_acc:.4f}")
    print(f"Non Homomorphic Accuracy: {non_homomorphic_accuracy:.4f}")
    print(f"Homomorphic Accuracy: {homomorphic_accuracy:.4f}")
    print(f"Difference Percentage: {difference:.2f}%")

    # bench: Measure: Sklearn accuracy = sklearn_acc
    # bench: Measure: Non Homomorphic Accuracy = non_homomorphic_accuracy
    # bench: Measure: Homomorphic Accuracy = homomorphic_accuracy
    # bench: Measure: Accuracy Difference Between Homomorphic and Non Homomorphic Implementation (%) = difference
    # bench: Alert: Accuracy Difference Between Homomorphic and Non Homomorphic Implementation (%) > 2


if __name__ == "__main__":
    main()

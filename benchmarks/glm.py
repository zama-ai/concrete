# bench: Full Target: Generalized Linear Model

from copy import deepcopy
from typing import Any, Dict

import numpy as np
from common import BENCHMARK_CONFIGURATION
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
from tqdm import tqdm

from concrete.quantization import QuantizedArray, QuantizedLinear, QuantizedModule
from concrete.quantization.quantized_activations import QuantizedActivation


class QuantizedExp(QuantizedActivation):
    """
    Quantized Exponential function

    This class will build a quantized lookup table for the exp function
    applied to input calibration data
    """

    def calibrate(self, x: np.ndarray):
        self.q_out = QuantizedArray(self.n_bits, np.exp(x))

    def __call__(self, q_input: QuantizedArray) -> QuantizedArray:
        """Process the forward pass of the exponential.

        Args:
            q_input (QuantizedArray): Quantized input.

        Returns:
            q_out (QuantizedArray): Quantized output.
        """

        quant_exp = np.exp(self.dequant_input(q_input))

        q_out = self.quant_output(quant_exp)
        return q_out


class QuantizedGLM(QuantizedModule):
    """
    Quantized Generalized Linear Model

    Building on top of QuantizedModule, this class will chain together a linear transformation
    and an inverse-link function
    """

    def __init__(self, n_bits, sklearn_model, calibration_data) -> None:
        self.n_bits = n_bits

        # We need to calibrate to a sufficiently low number of bits
        # so that the output of the Linear layer (w . x + b)
        # does not exceed 7 bits
        self.q_calibration_data = QuantizedArray(self.n_bits, calibration_data)

        # Quantize the weights and create the quantized linear layer
        q_weights = QuantizedArray(self.n_bits, np.expand_dims(sklearn_model.coef_, 1))
        q_bias = QuantizedArray(self.n_bits, sklearn_model.intercept_)
        q_layer = QuantizedLinear(self.n_bits, q_weights, q_bias)

        # Store quantized layers
        quant_layers_dict: Dict[str, Any] = {}

        # Calibrate the linear layer and obtain calibration_data for the next layers
        calibration_data = self._calibrate_and_store_layers_activation(
            "linear", q_layer, calibration_data, quant_layers_dict
        )

        # Add the inverse-link for inference.
        # This function needs to be quantized since it's computed in FHE.
        # However, we can use 7 bits of output since, in this case,
        # the result of the inverse-link is not processed by any further layers
        # Seven bits is the maximum precision but this could be lowered to improve speed
        # at the possible expense of higher deviance of the regressor
        q_exp = QuantizedExp(n_bits=7)

        # Now calibrate the inverse-link function with the linear layer's output data
        calibration_data = self._calibrate_and_store_layers_activation(
            "invlink", q_exp, calibration_data, quant_layers_dict
        )

        # Finally construct out Module using the quantized layers
        super().__init__(quant_layers_dict)

    def _calibrate_and_store_layers_activation(
        self, name, q_function, calibration_data, quant_layers_dict
    ):
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


def score_estimator(y_pred, y_gt, gt_weight):
    """Score an estimator on the test set."""

    y_pred = np.squeeze(y_pred)
    # Ignore non-positive predictions, as they are invalid for
    # the Poisson deviance. We want to issue a warning if for some reason
    # (e.g. FHE noise, bad quantization, user error), the regressor predictions are negative

    # Find all strictly positive values
    mask = y_pred > 0
    # If any non-positive values are found, issue a warning
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )

    # Compute the Poisson Deviance for all valid values
    dev = mean_poisson_deviance(
        y_gt[mask],
        y_pred[mask],
        sample_weight=gt_weight[mask],
    )
    print(f"mean Poisson deviance: {dev}")
    return dev


def score_sklearn_estimator(estimator, df_test):
    """A wrapper to score a sklearn pipeline on a dataframe"""
    return score_estimator(estimator.predict(df_test), df_test["Frequency"], df_test["Exposure"])


def score_concrete_glm_estimator(poisson_glm_pca, q_glm, df_test):
    """A wrapper to score QuantizedGLM on a dataframe, transforming the dataframe using
    a sklearn pipeline
    """
    test_data = poisson_glm_pca["pca"].transform(poisson_glm_pca["preprocessor"].transform(df_test))
    q_test_data = q_glm.quantize_input(test_data)
    y_pred = q_glm.forward_and_dequant(q_test_data)
    return score_estimator(y_pred, df_test["Frequency"], df_test["Exposure"])


def run_glm_benchmark():
    """
    This is our main benchmark function. It gets a dataset, trains a GLM model,
    then trains a GLM model on PCA reduced features, a QuantizedGLM model
    and finally compiles the QuantizedGLM to FHE. All models are evaluated and poisson deviance
    is computed to determine the increase in deviance from quantization and to verify
    that the FHE compiled model acheives the same deviance as the quantized model in the 'clear'
    """

    df, _ = fetch_openml(
        data_id=41214, as_frame=True, cache=True, data_home="~/.cache/sklean", return_X_y=True
    )
    df = df.head(50000)

    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    log_scale_transformer = make_pipeline(
        FunctionTransformer(np.log, validate=False), StandardScaler()
    )

    linear_model_preprocessor = ColumnTransformer(
        [
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("binned_numeric", KBinsDiscretizer(n_bins=10), ["VehAge", "DrivAge"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
            (
                "onehot_categorical",
                OneHotEncoder(sparse=False),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
        ],
        remainder="drop",
    )

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    df_calib, df_test = train_test_split(df_test, test_size=100, random_state=0)

    poisson_glm = Pipeline(
        [
            ("preprocessor", linear_model_preprocessor),
            ("regressor", PoissonRegressor(alpha=1e-12, max_iter=300)),
        ]
    )

    poisson_glm_pca = Pipeline(
        [
            ("preprocessor", linear_model_preprocessor),
            ("pca", PCA(n_components=15, whiten=True)),
            ("regressor", PoissonRegressor(alpha=1e-12, max_iter=300)),
        ]
    )

    poisson_glm.fit(df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"])

    poisson_glm_pca.fit(
        df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
    )

    # Let's check what prediction performance we lose due to PCA
    print("PoissonRegressor evaluation:")
    _ = score_sklearn_estimator(poisson_glm, df_test)
    print("PoissonRegressor+PCA evaluation:")
    _ = score_sklearn_estimator(poisson_glm_pca, df_test)

    # Now, get calibration data from the held out set
    calib_data = poisson_glm_pca["pca"].transform(
        poisson_glm_pca["preprocessor"].transform(df_calib)
    )

    # Let's see how performance decreases with bit-depth.
    # This is just a test of our quantized model, not in FHE
    for n_bits in [28, 16, 6, 5, 4, 3, 2]:
        q_glm = QuantizedGLM(n_bits, poisson_glm_pca["regressor"], calib_data)
        print(f"{n_bits}b Quantized PoissonRegressor evaluation:")
        score_concrete_glm_estimator(poisson_glm_pca, q_glm, df_test)

    q_glm = QuantizedGLM(2, poisson_glm_pca["regressor"], calib_data)
    dev_pca_quantized = score_concrete_glm_estimator(poisson_glm_pca, q_glm, df_test)
    test_data = poisson_glm_pca["pca"].transform(poisson_glm_pca["preprocessor"].transform(df_test))
    q_test_data = q_glm.quantize_input(test_data)

    # bench: Measure: Compilation Time (ms)
    engine = q_glm.compile(
        q_test_data,
        BENCHMARK_CONFIGURATION,
        show_mlir=False,
    )
    # bench: Measure: End

    y_pred_fhe = np.zeros((test_data.shape[0],), np.float32)
    for i, test_sample in enumerate(tqdm(q_test_data.qvalues)):
        # bench: Measure: Evaluation Time (ms)
        q_sample = np.expand_dims(test_sample, 1).transpose([1, 0]).astype(np.uint8)
        q_pred_fhe = engine.run(q_sample)
        y_pred_fhe[i] = q_glm.dequantize_output(q_pred_fhe)
        # bench: Measure: End

    dev_pca_quantized_fhe = score_estimator(y_pred_fhe, df_test["Frequency"], df_test["Exposure"])

    if dev_pca_quantized_fhe > 0.001:
        difference = abs(dev_pca_quantized - dev_pca_quantized_fhe) * 100 / dev_pca_quantized_fhe
    else:
        difference = 0

    print(f"Quantized deviance: {dev_pca_quantized}")
    print(f"FHE Quantized deviance: {dev_pca_quantized_fhe}")
    print(f"Percentage difference: {difference}%")

    # bench: Measure: Non Homomorphic Loss = dev_pca_quantized
    # bench: Measure: Homomorphic Loss = dev_pca_quantized_fhe
    # bench: Measure: Relative Loss Difference (%) = difference
    # bench: Alert: Relative Loss Difference (%) > 7.5


if __name__ == "__main__":
    run_glm_benchmark()

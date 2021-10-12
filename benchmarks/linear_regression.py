# Full Target: Linear Regression

# Disable line length warnings as we have a looooong metric...
# flake8: noqa: E501
# pylint: disable=C0301

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    x = np.array(
        [[69], [130], [110], [100], [145], [160], [185], [200], [80], [50]], dtype=np.float32
    )
    y = np.array([181, 325, 295, 268, 400, 420, 500, 520, 220, 120], dtype=np.float32)

    class Model:
        w = None
        b = None

        def fit(self, x, y):
            a = np.ones((x.shape[0], x.shape[1] + 1), dtype=np.float32)
            a[:, 1:] = x

            regularization_contribution = np.identity(x.shape[1] + 1, dtype=np.float32)
            regularization_contribution[0][0] = 0

            parameters = np.linalg.pinv(a.T @ a + regularization_contribution) @ a.T @ y

            self.b = parameters[0]
            self.w = parameters[1:].reshape(-1, 1)

            return self

        def evaluate(self, x):
            return x @ self.w + self.b

    model = Model().fit(x, y)

    class QuantizationParameters:
        def __init__(self, q, zp, n):
            self.q = q
            self.zp = zp
            self.n = n

    class QuantizedArray:
        def __init__(self, values, parameters):
            self.values = np.array(values)
            self.parameters = parameters

        @staticmethod
        def of(x, n):
            if not isinstance(x, np.ndarray):
                x = np.array(x)

            min_x = x.min()
            max_x = x.max()

            if min_x == max_x:

                if min_x == 0.0:
                    q_x = 1
                    zp_x = 0
                    x_q = np.zeros(x.shape, dtype=np.uint)

                elif min_x < 0.0:
                    q_x = abs(1 / min_x)
                    zp_x = -1
                    x_q = np.zeros(x.shape, dtype=np.uint)

                else:
                    q_x = 1 / min_x
                    zp_x = 0
                    x_q = np.ones(x.shape, dtype=np.uint)

            else:
                q_x = (2 ** n - 1) / (max_x - min_x)
                zp_x = int(round(min_x * q_x))
                x_q = ((q_x * x) - zp_x).round().astype(np.uint)

            return QuantizedArray(x_q, QuantizationParameters(q_x, zp_x, n))

        def dequantize(self):
            return (self.values.astype(np.float32) + float(self.parameters.zp)) / self.parameters.q

        def affine(self, w, b, min_y, max_y, n_y):
            x_q = self.values
            w_q = w.values
            b_q = b.values

            q_x = self.parameters.q
            q_w = w.parameters.q
            q_b = b.parameters.q

            zp_x = self.parameters.zp
            zp_w = w.parameters.zp
            zp_b = b.parameters.zp

            q_y = (2 ** n_y - 1) / (max_y - min_y)
            zp_y = int(round(min_y * q_y))

            y_q = (q_y / (q_x * q_w)) * (
                (x_q + zp_x) @ (w_q + zp_w) + (q_x * q_w / q_b) * (b_q + zp_b)
            )
            y_q -= min_y * q_y
            y_q = y_q.round().clip(0, 2 ** n_y - 1).astype(np.uint)

            return QuantizedArray(y_q, QuantizationParameters(q_y, zp_y, n_y))

    class QuantizedFunction:
        def __init__(self, table):
            self.table = table

        @staticmethod
        def of(f, input_bits, output_bits):
            domain = np.array(range(2 ** input_bits), dtype=np.uint)
            table = f(domain).round().clip(0, 2 ** output_bits - 1).astype(np.uint)
            return QuantizedFunction(table)

    parameter_bits = 1

    w_q = QuantizedArray.of(model.w, parameter_bits)
    b_q = QuantizedArray.of(model.b, parameter_bits)

    input_bits = 6

    x_q = QuantizedArray.of(x, input_bits)

    output_bits = 7

    min_y = y.min()
    max_y = y.max()

    n_y = output_bits
    q_y = (2 ** n_y - 1) / (max_y - min_y)
    zp_y = int(round(min_y * q_y))
    y_parameters = QuantizationParameters(q_y, zp_y, n_y)

    q_x = x_q.parameters.q
    q_w = w_q.parameters.q
    q_b = b_q.parameters.q

    zp_x = x_q.parameters.zp
    zp_w = w_q.parameters.zp
    zp_b = b_q.parameters.zp

    x_q = x_q.values
    w_q = w_q.values
    b_q = b_q.values

    c1 = q_y / (q_x * q_w)
    c2 = w_q + zp_w
    c3 = (q_x * q_w / q_b) * (b_q + zp_b)
    c4 = min_y * q_y

    f_q = QuantizedFunction.of(
        lambda intermediate: (c1 * (intermediate + c3)) - c4,
        input_bits + parameter_bits,
        output_bits,
    )

    table = hnp.LookupTable([int(entry) for entry in f_q.table])

    w_0 = int(c2.flatten()[0])

    def function_to_compile(x_0):
        return table[(x_0 + zp_x) * w_0]

    inputset = []
    for x_i in x_q:
        inputset.append((int(x_i[0]),))

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x_0": hnp.EncryptedScalar(hnp.UnsignedInteger(input_bits))},
        inputset,
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # Measure: End

    non_homomorphic_loss = 0
    homomorphic_loss = 0

    for i, (x_i, y_i) in enumerate(zip(x_q, y)):
        x_i = [int(value) for value in x_i]

        non_homomorphic_prediction = (
            QuantizedArray(x_i, QuantizationParameters(q_x, zp_x, input_bits))
            .affine(
                QuantizedArray.of(model.w, parameter_bits),
                QuantizedArray.of(model.b, parameter_bits),
                min_y,
                max_y,
                output_bits,
            )
            .dequantize()[0]
        )
        # Measure: Evaluation Time (ms)
        homomorphic_prediction = QuantizedArray(engine.run(*x_i), y_parameters).dequantize()
        # Measure: End

        non_homomorphic_loss += (non_homomorphic_prediction - y_i) ** 2
        homomorphic_loss += (homomorphic_prediction - y_i) ** 2

        print()

        print(f"input = {x[i][0]}")
        print(f"output = {y_i:.4f}")

        print(f"non homomorphic prediction = {non_homomorphic_prediction:.4f}")
        print(f"homomorphic prediction = {homomorphic_prediction:.4f}")

    non_homomorphic_loss /= len(y)
    homomorphic_loss /= len(y)
    difference = abs(homomorphic_loss - non_homomorphic_loss) * 100 / non_homomorphic_loss

    print()
    print(f"Non Homomorphic Loss: {non_homomorphic_loss:.4f}")
    print(f"Homomorphic Loss: {homomorphic_loss:.4f}")
    print(f"Relative Difference Percentage: {difference:.2f}%")

    # Measure: Non Homomorphic Loss = non_homomorphic_loss
    # Measure: Homomorphic Loss = homomorphic_loss
    # Measure: Relative Loss Difference Between Homomorphic and Non Homomorphic Implementation (%) = difference
    # Alert: Relative Loss Difference Between Homomorphic and Non Homomorphic Implementation (%) > 5


if __name__ == "__main__":
    main()

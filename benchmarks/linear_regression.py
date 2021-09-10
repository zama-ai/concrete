# Target: Linear Regression

import numpy as np

import concrete.numpy as hnp


def main():
    x = np.array([[130], [110], [100], [145], [160], [185], [200], [80], [50]], dtype=np.float32)
    y = np.array([325, 295, 268, 400, 420, 500, 520, 220, 120], dtype=np.float32)

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
        iter(inputset),
    )
    # Measure: End

    loss = 0
    for x_i, y_i in zip(x_q, y):
        x_i = [int(value) for value in x_i]

        # Measure: Evaluation Time (ms)
        prediction = QuantizedArray(engine.run(*x_i), y_parameters).dequantize()
        # Measure: End

        loss += (prediction - y_i) ** 2

    # Measure: Loss = loss / len(y)


if __name__ == "__main__":
    main()

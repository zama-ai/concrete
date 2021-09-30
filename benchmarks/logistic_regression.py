# Target: Logistic Regression

import numpy as np
import torch

import concrete.numpy as hnp


def main():
    x = torch.tensor([[1, 1], [1, 2], [2, 1], [4, 1], [3, 2], [4, 2]]).float()
    y = torch.tensor([[0], [0], [0], [1], [1], [1]]).float()

    class Model(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fc = torch.nn.Linear(n, 1)

        def forward(self, x):
            output = torch.sigmoid(self.fc(x))
            return output

    model = Model(x.shape[1])

    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    criterion = torch.nn.BCELoss()

    epochs = 1501
    for e in range(1, epochs + 1):
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        if e % 100 == 1 or e == epochs:
            print("Epoch:", e, "|", "Loss:", loss.item())

    w = np.array(model.fc.weight.flatten().tolist()).reshape((-1, 1))
    b = model.fc.bias.flatten().tolist()[0]

    x = x.detach().numpy()
    y = y.detach().numpy().flatten()

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
        def __init__(self, table, input_parameters=None, output_parameters=None):
            self.table = table
            self.input_parameters = input_parameters
            self.output_parameters = output_parameters

        @staticmethod
        def of(f, input_bits, output_bits):
            domain = np.array(range(2 ** input_bits), dtype=np.uint)
            table = f(domain).round().clip(0, 2 ** output_bits - 1).astype(np.uint)
            return QuantizedFunction(table)

        @staticmethod
        def plain(f, input_parameters, output_bits):
            n = input_parameters.n

            domain = np.array(range(2 ** n), dtype=np.uint)
            inputs = QuantizedArray(domain, input_parameters).dequantize()

            outputs = f(inputs)
            quantized_outputs = QuantizedArray.of(outputs, output_bits)

            table = quantized_outputs.values
            output_parameters = quantized_outputs.parameters

            return QuantizedFunction(table, input_parameters, output_parameters)

        def apply(self, x):
            assert x.parameters == self.input_parameters
            return QuantizedArray(self.table[x.values], self.output_parameters)

    parameter_bits = 1

    w_q = QuantizedArray.of(w, parameter_bits)
    b_q = QuantizedArray.of(b, parameter_bits)

    input_bits = 5

    x_q = QuantizedArray.of(x, input_bits)

    output_bits = 7

    intermediate = x @ w + b
    intermediate_q = x_q.affine(w_q, b_q, intermediate.min(), intermediate.max(), output_bits)

    sigmoid = QuantizedFunction.plain(
        lambda x: 1 / (1 + np.exp(-x)), intermediate_q.parameters, output_bits
    )

    y_q = sigmoid.apply(intermediate_q)
    y_parameters = y_q.parameters

    q_x = x_q.parameters.q
    q_w = w_q.parameters.q
    q_b = b_q.parameters.q
    q_intermediate = intermediate_q.parameters.q

    zp_x = x_q.parameters.zp
    zp_w = w_q.parameters.zp
    zp_b = b_q.parameters.zp

    x_q = x_q.values
    w_q = w_q.values
    b_q = b_q.values

    c1 = q_intermediate / (q_x * q_w)
    c2 = w_q + zp_w
    c3 = (q_x * q_w / q_b) * (b_q + zp_b)
    c4 = intermediate.min() * q_intermediate

    def f(x):
        values = ((c1 * (x + c3)) - c4).round().clip(0, 2 ** output_bits - 1).astype(np.uint)
        after_affine_q = QuantizedArray(values, intermediate_q.parameters)

        sigmoid = QuantizedFunction.plain(
            lambda x: 1 / (1 + np.exp(-x)),
            after_affine_q.parameters,
            output_bits,
        )
        y_q = sigmoid.apply(after_affine_q)

        return y_q.values

    f_q = QuantizedFunction.of(f, output_bits, output_bits)

    table = hnp.LookupTable([int(entry) for entry in f_q.table])

    w_0 = int(c2.flatten()[0])
    w_1 = int(c2.flatten()[1])

    def function_to_compile(x_0, x_1):
        return table[((x_0 + zp_x) * w_0) + ((x_1 + zp_x) * w_1)]

    inputset = []
    for x_i in x_q:
        inputset.append((int(x_i[0]), int(x_i[1])))

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {
            "x_0": hnp.EncryptedScalar(hnp.UnsignedInteger(input_bits)),
            "x_1": hnp.EncryptedScalar(hnp.UnsignedInteger(input_bits)),
        },
        inputset,
    )
    # Measure: End

    correct = 0
    for x_i, y_i in zip(x_q, y):
        x_i = [int(value) for value in x_i]

        # Measure: Evaluation Time (ms)
        prediction = round(QuantizedArray(engine.run(*x_i), y_parameters).dequantize())
        # Measure: End

        if prediction == y_i:
            correct += 1

    accuracy = (correct / len(y)) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Measure: Accuracy (%) = accuracy


if __name__ == "__main__":
    main()

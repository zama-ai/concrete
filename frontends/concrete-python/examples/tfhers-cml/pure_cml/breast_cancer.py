import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from concrete.ml.sklearn import LogisticRegression


# Train the model
X, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

model = LogisticRegression(n_bits=8)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (all-in-one API): {accuracy*100:.2f}%")

# Compile the model
model.compile(x_train, show_mlir=True)

# Run with individual APIs
x_test_q = model.quantize_input(x_test.astype(np.float32))
y_pred_q = model._inference(x_test_q)
y_pred_q_dequantized = model.dequantize_output(y_pred_q) > 0

# print(f"{x_test_q.shape}")
# print(f"{y_pred_q_dequantized.shape=}")
# print(f"{y_test.shape=}")

# print(f"{y_pred_q_dequantized=}")
# print(f"{y_test=}")

accuracy = accuracy_score(y_test, y_pred_q_dequantized)
print(f"Accuracy (individual APIs): {accuracy*100:.2f}%")


# Reimplement in pure numpy
def f(inputs):
    matrix = np.array([[127], [17], [-5], [-25], [-31], [-50], [-61], [-40], [-32], [-25], [-20], [70], [-4], [-39], [-24], [-28], [-30], [-26], [-25], [-24], [124], [-77], [-51], [-26], [-37], [-107], [-128], [-56], [-49], [-32]])
    res = np.matmul(inputs, matrix + 24) - 766
    return res

def g(inputs):
    matrix = np.array([[127], [17], [-5], [-25], [-31], [-50], [-61], [-40], [-32], [-25], [-20], [70], [-4], [-39], [-24], [-28], [-30], [-26], [-25], [-24], [124], [-77], [-51], [-26], [-37], [-107], [-128], [-56], [-49], [-32]])

    # 1536 comes from
    # print(model.output_quantizers[0].zero_point)

    res = np.matmul(inputs, matrix + 24) - 766 + 1536
    res_is_positif = res > 0
    return res_is_positif


def fall(all_inputs):

    d = all_inputs.shape[0]
    ans = np.zeros((d, 1))

    for i in range(d):
        inputs = all_inputs[i]
        ans_i = f(inputs)
        ans[i, :] = ans_i

    return ans

def gall(all_inputs):

    d = all_inputs.shape[0]
    ans = np.zeros((d, 1))

    for i in range(d):
        inputs = all_inputs[i]
        ans_i = g(inputs)
        ans[i, :] = ans_i

    return ans

# Run using quantization integer computation and dequantization (for embedding within concrete-python)
x_test_bcm = model.quantize_input(x_test.astype(np.float32))
if False:
    y_pred_bcm = fall(x_test_bcm)
    y_pred_bcm_dequantized = model.dequantize_output(y_pred_bcm) > 0
else:
    y_pred_bcm_dequantized = gall(x_test_bcm)

accuracy = accuracy_score(y_test, y_pred_bcm_dequantized)
print(f"Accuracy (reimplementation): {accuracy*100:.2f}%")



import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from concrete.ml.sklearn import LogisticRegression


# Train the model
X, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1)

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

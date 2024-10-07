import pandas as pd
import numpy as np
import onnx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from concrete.ml.sklearn import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Load dataset
df = pd.read_csv('insurance.csv')

# Encode categorical variables
categorical_columns = ['sex', 'smoker', 'region']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Log-transform the target variable
df['charges'] = np.log(df['charges'])

# Define features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Pure scikit-learn
model = SklearnLinearRegression()
model.fit(X_train, y_train)

# Predict in the clear
y_pred_sklearn = model.predict(X_test)

# Initialize the Concrete-ML Linear Regression model
model = LinearRegression(n_bits=8)
model.fit(X_train, y_train)

# Predict in the clear
y_pred_clear = model.predict(X_test)

# Compile the model for FHE
model.compile(X_train, show_mlir=True)

# Predict using FHE
y_pred_fhe = model.predict(X_test, fhe="execute")

# Evaluate the model
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
mse_clear = mean_squared_error(y_test, y_pred_clear)
r2_clear = r2_score(y_test, y_pred_clear)
mse_fhe = mean_squared_error(y_test, y_pred_fhe)
r2_fhe = r2_score(y_test, y_pred_fhe)

print(f'Mean Squared Error (Scikit-learn): {mse_sklearn}')
print(f'R² Score (Scikit-learn): {r2_sklearn}')
print(f'Mean Squared Error (Clear): {mse_clear}')
print(f'R² Score (Clear): {r2_clear}')
print(f'Mean Squared Error (FHE): {mse_fhe}')
print(f'R² Score (FHE): {r2_fhe}')
print(f"Similarity between clear and FHE predictions: {int((y_pred_fhe == y_pred_clear).mean() * 100)}%")


# Run using quantization integer computation and dequantization (for embedding within concrete-python)
X_test_q = model.quantize_input(X_test.to_numpy().astype(np.float32))

y_pred_q = model._inference(X_test_q)  # That's what you should use in concrete-python function

y_pred_q_dequantized = model.dequantize_output(y_pred_q)

mse_fhe_q = mean_squared_error(y_test, y_pred_q_dequantized)
r2_fhe_q = r2_score(y_test, y_pred_q_dequantized)

print(f'Mean Squared Error (FHE Quantized): {mse_fhe_q}')
print(f'R² Score (FHE Quantized): {r2_fhe_q}')

# Do one example
X_test_one = X_test[1:2]
print(f"{X_test_one=}")

X_test_q_one = model.quantize_input(X_test_one.to_numpy().astype(np.float32))
print(f"{X_test_q_one=}")

y_pred_q_one = model._inference(X_test_q_one)  # That's what you should use in concrete-python function
print(f"{y_pred_q_one=}")

y_pred_q_dequantized_one = model.dequantize_output(y_pred_q_one)
print(f"{y_pred_q_dequantized_one=}")

# Reimplement in pure numpy
def f(inputs):
    matrix = np.array([-101, -104, -91, -117, 127, -114, -128, -126])
    res = np.matmul(inputs, matrix + 106) - 20584
    return res

# And check
print(f"{f(np.array([ 115,  -16, -128, -128, -128, -128, -128, -124]))=}")
print(f"{f(np.array([  -4,  -18, -120, -124, -128, -128, -128, -128]))=}")

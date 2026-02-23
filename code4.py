# ==============================
# Linear & Polynomial Regression
# ==============================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# 1. Simple Linear Regression
# --------------------------------
X_train_simple = X_train[['MedInc']]
X_test_simple = X_test[['MedInc']]

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)

# --------------------------------
# 2. Multiple Linear Regression
# --------------------------------
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)

# --------------------------------
# 3. Polynomial Regression (degree=2)
# --------------------------------
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_simple)
X_test_poly = poly.transform(X_test_simple)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# --------------------------------
# Evaluation Function
# --------------------------------
def evaluate(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred))
    )

r2_s, mae_s, rmse_s = evaluate(y_test, y_pred_simple)
r2_m, mae_m, rmse_m = evaluate(y_test, y_pred_multi)
r2_p, mae_p, rmse_p = evaluate(y_test, y_pred_poly)

# --------------------------------
# Results
# --------------------------------
results = pd.DataFrame({
    "Model": ["Simple Linear", "Multiple Linear", "Polynomial (deg=2)"],
    "R2 Score": [r2_s, r2_m, r2_p],
    "MAE": [mae_s, mae_m, mae_p],
    "RMSE": [rmse_s, rmse_m, rmse_p]
})

print("\nModel Comparison:\n")
print(results)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def regression():
    X = np.array([[1],[2],[3],[4],[5]])
    y = np.array([2,4,5,4,5])

    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    print("MAE:", mean_absolute_error(y, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, pred)))

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model2 = LinearRegression().fit(X_poly, y)
    print("Polynomial Done")

regression()

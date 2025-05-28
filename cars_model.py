import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

num_cars = 5000

milage = np.random.rand(num_cars, 1) * 200000 + 10000
age = np.random.rand(num_cars, 1) * 15 + 1
engine_size = np.random.rand(num_cars, 1) * 4.5 + 1.5
is_luxury = np.random.choice([0, 1], size=(num_cars, 1), p=[0.7, 0.3])

true_intercept = 50000
coef_milage = -0.20
coef_age = 2000
coef_engine_size = 4000
coef_luxury = 20000

car_price = (
    true_intercept +
    coef_milage * milage +
    coef_age * age +
    coef_engine_size * engine_size +
    coef_luxury * is_luxury +
    np.random.randn(num_cars, 1) * 7000
)

car_price[car_price < 7000] = 7000

X = np.hstack((milage, age, engine_size, is_luxury))
y = car_price
df = pd.DataFrame(X, columns=["Milage", "Age", "Engine", "Luxury"])
df["Price"] = y
df.head()
df.describe()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

mse = mean_squared_error(y_test, y_predict)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, alpha=.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

cv_score = cross_val_score(model, X, y, cv=20, scoring="neg_mean_squared_error")
cv_score = -cv_score

mse_cv_sqrt = np.sqrt(cv_score)
avg_mse_cv_sqrt = np.sqrt(np.mean(cv_score))

print(f"CV MSE: { mse_cv_sqrt}")
print(f"Average CV MSE: { avg_mse_cv_sqrt: .2f}")





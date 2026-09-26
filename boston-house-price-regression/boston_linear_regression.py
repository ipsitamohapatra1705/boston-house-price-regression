"""
Simple Linear Regression on the Boston Housing dataset.

Predicts median house price (MEDV, in $1000s) from the average number
of rooms per dwelling (RM).

Note: sklearn.datasets.load_boston was removed in scikit-learn 1.2, so the
data is loaded directly from the original CMU source.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the data
DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(DATA_URL, sep=r"\s+", skiprows=22, header=None)

# The raw file stores each record across two lines; stitch them together
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]
df = pd.DataFrame(data, columns=columns)
df["MEDV"] = target

print(df.head())
print(f"\nDataset shape: {df.shape}")

# 2. Pick one feature (simple = one independent variable)
X = df[["RM"]]   # average rooms per dwelling
y = df["MEDV"]   # median home value in $1000s

# 3. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nIntercept: {model.intercept_:.3f}")
print(f"Slope (coefficient for RM): {model.coef_[0]:.3f}")

# 5. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R^2:  {r2:.3f}")

# 6. Visualize
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="steelblue", alpha=0.7, label="Actual")
plt.plot(X_test, y_pred, color="crimson", linewidth=2, label="Regression line")
plt.xlabel("Average number of rooms (RM)")
plt.ylabel("Median house price (MEDV, $1000s)")
plt.title("Boston Housing: Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("regression_plot.png", dpi=150)
plt.show()

# 7. Predict for a new house
new_house = pd.DataFrame({"RM": [6.5]})
predicted_price = model.predict(new_house)[0]
print(f"\nPredicted price for a 6.5-room house: ${predicted_price * 1000:,.0f}")

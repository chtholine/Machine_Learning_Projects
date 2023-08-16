import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=";")

# Data analysis
print(data.head())
print(data.info())
print(data.isnull().sum())

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Feature engineering
X["total_acidity"] = X["fixed acidity"] + X["volatile acidity"] + X["citric acid"]
X["sulphate_to_ph"] = X["sulphates"] / X["pH"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parts division
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

# Base model training with default hyperparameters
base_model = SGDRegressor()
base_model.fit(X_train, y_train)


# Base model score
base_pred = base_model.predict(X_val)
base_mse = mean_squared_error(y_val, base_pred)
r2 = r2_score(y_val, base_pred)
print("Mean Squared Error (base model):", base_mse)
print(f"R^2 Score(base model): {r2}")

# Hyperparameters search
params = {"fit_intercept": [True, False], "penalty": ["l1", "l2"]}
grid_search = GridSearchCV(SGDRegressor(), params, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Model score with optimal hyperparameters
best_pred = best_model.predict(X_val)
best_mse = mean_squared_error(y_val, best_pred)
best_r2 = r2_score(y_val, best_pred)
print("Mean Squared Error (best model):", best_mse)
print("R-squared (best model):", best_r2)
best_pred_test = best_model.predict(X_test)
best_mse_test = mean_squared_error(y_test, best_pred_test)
best_r2_test = r2_score(y_test, best_pred_test)
print("Mean Squared Error (best model. test):", best_mse_test)
print("R-squared (best model. test):", best_r2_test)

# Feature visualization
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# Fixed acidity
axes[0, 0].hist(data["fixed acidity"], bins=30, color="lightblue", edgecolor="black")
axes[0, 0].set_xlabel("Fixed Acidity")
axes[0, 0].set_ylabel("Frequency")

# Volatile acidity
axes[0, 1].hist(data["volatile acidity"], bins=30, color="lightblue", edgecolor="black")
axes[0, 1].set_xlabel("Volatile Acidity")
axes[0, 1].set_ylabel("Frequency")

# Citric acid
axes[0, 2].hist(data["citric acid"], bins=30, color="lightblue", edgecolor="black")
axes[0, 2].set_xlabel("Citric Acid")
axes[0, 2].set_ylabel("Frequency")

# Residual sugar
axes[1, 0].hist(data["residual sugar"], bins=30, color="lightblue", edgecolor="black")
axes[1, 0].set_xlabel("Residual Sugar")
axes[1, 0].set_ylabel("Frequency")

# Chlorides
axes[1, 1].hist(data["chlorides"], bins=30, color="lightblue", edgecolor="black")
axes[1, 1].set_xlabel("Chlorides")
axes[1, 1].set_ylabel("Frequency")

# Free sulfur dioxide
axes[1, 2].hist(
    data["free sulfur dioxide"], bins=30, color="lightblue", edgecolor="black"
)
axes[1, 2].set_xlabel("Free Sulfur Dioxide")
axes[1, 2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

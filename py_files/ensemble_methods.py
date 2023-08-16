import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("../datasets/auction_data.csv")

data["total_capacity"] = (
    data["process.b1.capacity"]
    + data["process.b2.capacity"]
    + data["process.b3.capacity"]
    + data["process.b4.capacity"]
)
data.drop(
    columns=[
        "process.b1.capacity",
        "process.b2.capacity",
        "process.b3.capacity",
        "process.b4.capacity",
    ],
    inplace=True,
)

scaler = StandardScaler()
X = data.drop(columns=["verification.result", "verification.time"])
y = data["verification.result"]
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# Random Forest base model training
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_pred)
print("Random Forest accuracy:", rf_accuracy)

# Gradient Boosting base model training
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)
gb_accuracy = accuracy_score(y_val, gb_pred)
print("Gradient Boosting accuracy:", gb_accuracy)

# Bagging base model training
bg_model = BaggingClassifier(random_state=42)
bg_model.fit(X_train, y_train)
bg_pred = bg_model.predict(X_val)
bg_accuracy = accuracy_score(y_val, bg_pred)
print("Bagging accuracy:", bg_accuracy)

rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

gb_params = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.1, 0.05, 0.01],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

bg_params = {
    "n_estimators": [50, 100, 200],
    "max_samples": [0.5, 1.0],
    "max_features": [0.5, 1.0],
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), rf_params, n_iter=10, cv=5, random_state=42
)
rf_random.fit(X_train, y_train)

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    n_iter=10,
    cv=5,
    random_state=42,
)
gb_random.fit(X_train, y_train)

bg_random = RandomizedSearchCV(
    BaggingClassifier(random_state=42), bg_params, n_iter=10, cv=5, random_state=42
)
bg_random.fit(X_train, y_train)

print("\nBest parameters for Random Forest:", rf_random.best_params_)
print("Best parameters for Gradient Boosting:", gb_random.best_params_)
print("Best parameters for Bagging:", bg_random.best_params_)


rf_best_model = rf_random.best_estimator_
rf_val_pred = rf_best_model.predict(X_val)
rf_val_accuracy = accuracy_score(y_val, rf_val_pred)

gb_best_model = gb_random.best_estimator_
gb_val_pred = gb_best_model.predict(X_val)
gb_val_accuracy = accuracy_score(y_val, gb_val_pred)

bg_best_model = bg_random.best_estimator_
bg_val_pred = bg_best_model.predict(X_val)
bg_val_accuracy = accuracy_score(y_val, bg_val_pred)

print("\nRandom Forest accuracy on the validation set:", rf_val_accuracy)
print("Gradient Boosting accuracy on the validation set:", gb_val_accuracy)
print("Bagging accuracy on the validation set:", bg_val_accuracy)

rf_test_pred = rf_best_model.predict(X_test)
gb_test_pred = gb_best_model.predict(X_test)
bg_test_pred = bg_best_model.predict(X_test)

rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
gb_test_accuracy = accuracy_score(y_test, gb_test_pred)
bg_test_accuracy = accuracy_score(y_test, bg_test_pred)

print("\nRandom Forest accuracy on the test set:", rf_test_accuracy)
print("Gradient Boosting accuracy on the test set:", gb_test_accuracy)
print("Bagging accuracy on the test set:", bg_test_accuracy)


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, rf_test_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

estimators = [("rf", rf_best_model), ("gb", gb_best_model), ("bg", bg_best_model)]

stacking_classifier = StackingClassifier(
    estimators=estimators, final_estimator=DecisionTreeClassifier()
)
stacking_classifier.fit(X_train, y_train)

stacking_val_pred = stacking_classifier.predict(X_val)
stacking_val_accuracy = accuracy_score(y_val, stacking_val_pred)
print("\nStacking accuracy on the validation set:", stacking_val_accuracy)

stacking_test_pred = stacking_classifier.predict(X_test)
stacking_test_accuracy = accuracy_score(y_test, stacking_test_pred)
print("Stacking accuracy on the test set:", stacking_test_accuracy)

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, gb_test_pred), annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, bg_test_pred), annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - Bagging")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, stacking_test_pred), annot=True, fmt="d", cmap="Purples"
)
plt.title("Confusion Matrix - Stacking (Test)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

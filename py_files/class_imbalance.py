import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
data = pd.read_csv(url, names=names)

print(data.head())
print(data.isnull().sum())

# Розділення на фічі і мітки
X = data.drop(["Id", "Type"], axis=1)
y = data["Type"]

X["Light_elements"] = X["Na"] + X["Mg"]
X["Heavy_elements"] = X["Al"] + X["Si"] + X["K"] + X["Ca"] + X["Ba"] + X["Fe"]
X["Total_elements"] = (
    X["Na"] + X["Mg"] + X["Al"] + X["Si"] + X["K"] + X["Ca"] + X["Ba"] + X["Fe"]
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# RandomOverSampler для збільшення кількості зразків менших класів.
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

model = LogisticRegression(solver="saga", penalty="l2", max_iter=1000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    LogisticRegression(random_state=42), param_grid, cv=5, n_jobs=-1
)
grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_

# Оцінка результатів
y_pred_val = best_model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
confusion_val = confusion_matrix(y_val, y_pred_val)
classification_report_val = classification_report(y_val, y_pred_val)

print(
    f"""Validation Accuracy:"
{accuracy_val}

"Confusion Matrix:"
{confusion_val}

"Classification Report:"
{classification_report_val}
"""
)

print("Кількість зразків у кожному класі перед вирівнюванням:")
unique_classes, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")

plt.figure(figsize=(8, 5))
plt.bar(unique_classes, counts)
plt.xlabel("Клас")
plt.ylabel("Кількість зразків")
plt.title("Кількість зразків у кожному класі перед вирівнюванням")
plt.xticks(unique_classes)
plt.show()

print("Кількість зразків у кожному класі після вирівнювання:")
unique_classes, counts = np.unique(y_train_resampled, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")

plt.figure(figsize=(8, 5))
plt.bar(unique_classes, counts)
plt.xlabel("Клас")
plt.ylabel("Кількість зразків")
plt.title("Кількість зразків у кожному класі після вирівнюванням")
plt.xticks(unique_classes)
plt.show()

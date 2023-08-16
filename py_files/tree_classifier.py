import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
column_names = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
]
data = pd.read_csv(url, header=None, names=column_names, na_values="?")
data = pd.get_dummies(
    data, columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
)

print(data.head())
print(data.info())
print(data.isnull().sum())

# unique values in categorical features
print(data.select_dtypes(include="object").nunique())

# feature engineering
data["TotalIncome"] = data["A11"] + data["A15"]
scaler = StandardScaler()
data[["A2", "A3", "A8", "A11", "A14", "A15", "TotalIncome"]] = scaler.fit_transform(
    data[["A2", "A3", "A8", "A11", "A14", "A15", "TotalIncome"]]
)
X = data.drop("A16", axis=1)
y = data["A16"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# pipeline for filling in missing values and scaling features
pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())

# applying pipeline
X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)
X_test = pipeline.transform(X_test)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_val_pred = dt_classifier.predict(X_val)
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Accuracy on the validation set: {accuracy_val:.4f}")

param_grid = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=dt_classifier, param_grid=param_grid, cv=5, n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"The best hyperparameters:\n{grid_search.best_params_}")
print(f"Accuracy on the validation set:\n{grid_search.best_score_}")
y_test_pred = grid_search.best_estimator_.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on the test set: {accuracy_test:.4f}")
print(f"Classification report:\n{classification_report(y_test, y_test_pred)}")

y_test_pred = dt_classifier.predict(X_test)

confusion_matrix = pd.crosstab(
    y_test, y_test_pred, rownames=["Actual"], colnames=["Predicted"]
)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.show()

feature_importance = dt_classifier.feature_importances_
features = X.columns
sorted_idx = feature_importance.argsort()

num_features = 10

plt.figure(figsize=(10, 6))
plt.barh(
    range(num_features), feature_importance[sorted_idx][-num_features:], align="center"
)
plt.yticks(range(num_features), [features[i] for i in sorted_idx][-num_features:])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.show()

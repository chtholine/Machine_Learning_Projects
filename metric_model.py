import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(url, column_names):
    data = pd.read_csv(url, names=column_names)
    data["Volume"] = (
        (4 / 3) * np.pi * (data["Length"] * data["Diameter"] * data["Height"])
    )
    label_encoder = LabelEncoder()
    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    X = data.drop(["Rings"], axis=1)
    y_age = data["Rings"]
    y_sex = data["Sex"]
    return X, y_age, y_sex


def split_data(X, y_age, y_sex):
    X_scaled = StandardScaler().fit_transform(X)
    (
        X_train,
        X_test,
        y_age_train,
        y_age_test,
        y_sex_train,
        y_sex_test,
    ) = train_test_split(X_scaled, y_age, y_sex, test_size=0.2, random_state=42)
    X_train, X_val, y_age_train, y_age_val, y_sex_train, y_sex_val = train_test_split(
        X_train, y_age_train, y_sex_train, test_size=0.2, random_state=42
    )
    return (
        X_train,
        X_val,
        X_test,
        y_age_train,
        y_age_val,
        y_age_test,
        y_sex_train,
        y_sex_val,
        y_sex_test,
    )


def train_model(X_train, y_train, param_grid):
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model


def evaluate_model(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    return accuracy_score(y_val, y_val_pred)


# Load and preprocess the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]
X, y_age, y_sex = preprocess_data(url, column_names)

# Split the data
(
    X_train,
    X_val,
    X_test,
    y_age_train,
    y_age_val,
    y_age_test,
    y_sex_train,
    y_sex_val,
    y_sex_test,
) = split_data(X, y_age, y_sex)

# Train and evaluate age classification model
age_param_grid = {"n_neighbors": [3, 5, 7, 9]}
age_model = train_model(X_train, y_age_train, age_param_grid)
age_accuracy = evaluate_model(age_model, X_val, y_age_val)
print(f"Accuracy for Age Classification (Best Model): {age_accuracy}")

# Train and evaluate sex classification model
sex_param_grid = {"n_neighbors": [3, 5, 7, 9]}
sex_model = train_model(X_train, y_sex_train, sex_param_grid)
sex_accuracy = evaluate_model(sex_model, X_val, y_sex_val)
print(f"Accuracy for Sex Classification (Best Model): {sex_accuracy}")

# Visualize the distribution of ages
plt.figure(figsize=(8, 6))
sns.histplot(y_age, bins=30, kde=True)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Visualize the number of abalones by sex
data = pd.read_csv(url, names=column_names)
sex_counts = data['Sex'].value_counts()
plt.bar(sex_counts.index, sex_counts.values)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Abalone Count by Sex')
plt.xticks(range(len(sex_counts.index)), sex_counts.index)
plt.show()



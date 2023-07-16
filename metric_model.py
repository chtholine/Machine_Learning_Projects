import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
data = pd.read_csv(url, names=column_names)

# Фіча інжиніринг
data['Volume'] = (4/3) * np.pi * (data['Length'] * data['Diameter'] * data['Height'])

# Перетворення категоріальної змінної 'Sex' у числовий
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Поділ датасету на ознаки (X) та цільові змінні (y_age, y_sex)
X = data.drop(['Rings'], axis=1)
y_age = data['Rings']
y_sex = data['Sex']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Поділ датасету на тренувальну, валідаційну та тестову вибірки
X_train, X_test, y_age_train, y_age_test, y_sex_train, y_sex_test = train_test_split(
    X_scaled, y_age, y_sex, test_size=0.2, random_state=42
)

# Поділ тренувальної вибірки на тренувальну та валідаційну частини
X_train, X_val, y_age_train, y_age_val, y_sex_train, y_sex_val = train_test_split(
    X_train, y_age_train, y_sex_train, test_size=0.2, random_state=42
)

# Тренування базової моделі для класифікації віку з дефолтними гіперпараметрами
age_model = KNeighborsClassifier()
age_model.fit(X_train, y_age_train)

# Прогнозування віку на валідаційних даних
y_age_val_pred = age_model.predict(X_val)

# Оцінка точності базової моделі для класифікації віку
age_accuracy = accuracy_score(y_age_val, y_age_val_pred)
print(f'Accuracy for Age Classification (Base Model): {age_accuracy}')

# Підбір гіперпараметрів для моделі класифікації віку
param_grid = {'n_neighbors': [3, 5, 7, 9]}
age_model_grid = KNeighborsClassifier()
age_grid_search = GridSearchCV(age_model_grid, param_grid, cv=5)
age_grid_search.fit(X_train, y_age_train)

# Найкращі гіперпараметри для моделі класифікації віку
best_age_params = age_grid_search.best_params_

# Навчання моделі з підібраними гіперпараметрами для класифікації віку
age_model_best = KNeighborsClassifier(**best_age_params)
age_model_best.fit(X_train, y_age_train)

# Прогнозування віку на валідаційних даних з підібраними гіперпараметрами
y_age_val_pred_best = age_model_best.predict(X_val)

# Оцінка точності моделі для класифікації віку з підібраними гіперпараметрами
age_accuracy_best = accuracy_score(y_age_val, y_age_val_pred_best)
print(f'Accuracy for Age Classification (Tuned Model): {age_accuracy_best}')

# Тренування базової моделі для класифікації статі з дефолтними гіперпараметрами
sex_model = KNeighborsClassifier()
sex_model.fit(X_train, y_sex_train)

# Прогнозування статі на валідаційних даних
y_sex_val_pred = sex_model.predict(X_val)

# Оцінка точності базової моделі для класифікації статі
sex_accuracy = accuracy_score(y_sex_val, y_sex_val_pred)
print(f'Accuracy for Sex Classification (Base Model): {sex_accuracy}')

# Підбір гіперпараметрів для моделі класифікації статі
param_grid = {'n_neighbors': [3, 5, 7, 9]}
sex_model_grid = KNeighborsClassifier()
sex_grid_search = GridSearchCV(sex_model_grid, param_grid, cv=5)
sex_grid_search.fit(X_train, y_sex_train)

# Найкращі гіперпараметри для моделі класифікації статі
best_sex_params = sex_grid_search.best_params_

# Навчання моделі з підібраними гіперпараметрами для класифікації статі
sex_model_best = KNeighborsClassifier(**best_sex_params)
sex_model_best.fit(X_train, y_sex_train)

# Прогнозування статі на валідаційних даних з підібраними гіперпараметрами
y_sex_val_pred_best = sex_model_best.predict(X_val)

# Оцінка точності моделі для класифікації статі з підібраними гіперпараметрами
sex_accuracy_best = accuracy_score(y_sex_val, y_sex_val_pred_best)
print(f'Accuracy for Sex Classification (Tuned Model): {sex_accuracy_best}')

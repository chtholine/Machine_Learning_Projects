import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Завантаження даних
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
data = pd.read_csv(url, names=column_names)

# Первинний аналіз даних
print(data.head())
print(data.info())
print(data.isnull().sum())

# Перетворення категоріальної змінної 'Sex' у числовий
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Поділ датасету на ознаки (X) та цільові змінні (y_age, y_sex)
X = data.drop(['Rings'], axis=1)
y_age = data['Rings']
y_sex = data['Sex']

# Масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Поділ датасету на тренувальну та тестову вибірки
X_train, X_test, y_age_train, y_age_test, y_sex_train, y_sex_test = train_test_split(
    X_scaled, y_age, y_sex, test_size=0.2, random_state=42
)

# Тренування моделі для класифікації віку
age_model = KNeighborsClassifier(n_neighbors=5)
age_model.fit(X_train, y_age_train)

# Прогнозування віку на тестових даних
y_age_pred = age_model.predict(X_test)

# Оцінка точності моделі для класифікації віку
age_accuracy = accuracy_score(y_age_test, y_age_pred)
print(f'Accuracy for Age Classification: {age_accuracy}')

# Тренування моделі для класифікації статі
sex_model = KNeighborsClassifier(n_neighbors=5)
sex_model.fit(X_train, y_sex_train)

# Прогнозування статі на тестових даних
y_sex_pred = sex_model.predict(X_test)

# Оцінка точності моделі для класифікації статі
sex_accuracy = accuracy_score(y_sex_test, y_sex_pred)
print(f'Accuracy for Sex Classification: {sex_accuracy}')

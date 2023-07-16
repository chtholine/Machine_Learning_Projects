from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.data
y = digits.target

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

C_values = [0.1, 1, 10]
kernel_values = ['linear', 'poly', 'rbf']

param_grid = {'C': C_values, 'kernel': kernel_values}

model = SVC(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train_val, y_train_val)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Найкращі гіперпараметри |", best_parameters)
print("Точність |", best_accuracy)

# Visualization
results = grid_search.cv_results_
sorted_indices = results['rank_test_score'].argsort()
sorted_hyperparams = [f"C: {results['param_C'][i]}, kernel: {results['param_kernel'][i]}" for i in sorted_indices]
sorted_accuracies = results['mean_test_score'][sorted_indices]

plt.figure(figsize=(10, 6))
plt.bar(sorted_hyperparams, sorted_accuracies)
plt.xticks(rotation=45)
plt.xlabel('Гіперпараметри')
plt.ylabel('Точність')
plt.title('Продуктивність комбінацій гіперпараметрів')
plt.ylim(0.9, 1)
plt.show()

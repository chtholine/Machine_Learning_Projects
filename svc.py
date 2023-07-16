import itertools
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.data
y = digits.target

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

C_values = [0.1, 1, 10]
kernel_values = ['linear', 'poly', 'rbf']

best_accuracy = 0
best_parameters = {}

hyperparams = []
accuracies = []

for C, kernel in itertools.product(C_values, kernel_values):
    model = SVC(C=C, kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    hyperparams.append(f"C: {C}, kernel: {kernel}")
    accuracies.append(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_parameters = {'C': C, 'kernel': kernel}
    print(f"C: {C}, kernel: {kernel} |", accuracy)
print("\nНайкращі гіперпараметри |", best_parameters, "\nТочність |", best_accuracy)

sorted_results = sorted(zip(hyperparams, accuracies), key=lambda x: x[1], reverse=True)
sorted_hyperparams, sorted_accuracies = zip(*sorted_results)

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(sorted_hyperparams, sorted_accuracies)
plt.xticks(rotation=45)
plt.xlabel('Гіперпараметри')
plt.ylabel('Точність')
plt.title('Продуктивність комбінацій гіперпараметрів')
plt.ylim(0.9, 1)
plt.show()

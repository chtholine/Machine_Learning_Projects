import numpy as np

# Task 1
dataset = np.genfromtxt('datasets/iris.csv', delimiter=',')
target = dataset[:, -1]
dataset = np.delete(dataset, -1, axis=1)
print(f"Task #1:\n{dataset}\n")

# Task 2
if dataset.ndim == 2:
    print(f"Task #2:\nDataset is 2D\n")
else:
    dataset = dataset.reshape(-1, 4)
    print(f"Task #2\nDataset is reshaped to 2D\n{dataset}\n")

# Task 3
column1 = dataset[:, 0]
mean = np.mean(column1)
median = np.median(column1)
std = np.std(column1)
print(f"Task #3:\nMean: {mean}\nMedian: {median}\nStandard deviation: {std}\n")

# Task 4
np.random.seed(100)
indices = np.random.choice(dataset.size, size=20, replace=False)
dataset.flat[indices] = np.nan
print(f"Task #4. Insert random np.nan:\n{dataset}\n")

# Task 5
nan_positions = np.argwhere(np.isnan(dataset[:, 0]))
print(f"Task #5. np.nan positions:\n{nan_positions}\n")

# Task 6
filtered_dataset = dataset[(dataset[:, 2] > 1.5) & (dataset[:, 0] < 5.0)]
print(f"Task #6. Filtered dataset:\n{filtered_dataset}\n")

# Task 7
dataset[np.isnan(dataset)] = 0
print(f"Task #7:\n{dataset}\n")

# Task 8
unique_values, value_counts = np.unique(dataset, return_counts=True)
print("Task #8:")
for value, count in zip(unique_values, value_counts):
    print(f"Unique value/count: {value}/{count}\n")

# Task 9
split_datasets = np.hsplit(dataset, 2)
print(f"Task #9. Horizontally split dataset:\n{split_datasets}\n")

# Task 10
split_datasets[0] = split_datasets[0][np.argsort(split_datasets[0][:, 0])]
split_datasets[1] = split_datasets[1][np.argsort(split_datasets[1][:, 0])[::-1]]
print(f"Task #10 Sorted datasets: \n{split_datasets[0]}\n{split_datasets[1]}\n")

# Task 11
merged_dataset = np.hstack(split_datasets)
print(f"Task #11. Merged dataset:\n{merged_dataset}\n")

# Task 12
unique_values, value_counts = np.unique(merged_dataset.flatten(), return_counts=True)
most_common_value = unique_values[np.argmax(value_counts)]
most_common_count = np.max(value_counts)
print(f"Task #12. Most common value/count: {most_common_value}/{most_common_count}\n")

# Task 13
def column_function(column):
    mean = np.mean(column)
    column[column < mean] *= 2
    column[column >= mean] /= 4
    return column

# Task 14
merged_dataset[:, 2] = column_function(merged_dataset[:, 2])
print(f"Task #14. Applied function to the dataset:\n{merged_dataset}\n")

# Task 15
merged_dataset = np.sort(merged_dataset, axis=0)
print(f"Task #15. Sorted columns:\n{merged_dataset}\n")

# Task 16
cumulative_col_sum = np.cumsum(merged_dataset, axis=0)
print(f"Task #16. Cumulative sum of columns:\n{cumulative_col_sum[-1]}\n")

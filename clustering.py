import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset and skip the header row
data = pd.read_csv("SouthGermanCredit.asc", sep=" ")
print(data.head())
print(data.info())

# Feature engineering: Adding a new feature 'total_assets' as a sum of 'verm', 'sparkont', and 'wertkred'
data["total_assets"] = data["verm"] + data["sparkont"] + data["weitkred"]

# Extract features and target variable
X = data.drop("kredit", axis=1)
y = data["kredit"]

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training, validation, and test sets
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Applying PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Applying LDA for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

# Plotting PCA and LDA projections
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.title("PCA Projection")

plt.subplot(1, 2, 2)
plt.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap="viridis")
plt.title("LDA Projection")
plt.show()


# Clustering using KMeans
def evaluate_kmeans(X, n_clusters):
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    )  # Explicitly set n_init
    kmeans.fit(X)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    return labels, inertia


# Find the optimal number of clusters using elbow method
inertia_values = []
for i in range(1, 11):
    labels, inertia = evaluate_kmeans(X_scaled, i)
    inertia_values.append(inertia)

plt.plot(range(1, 11), inertia_values, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Cluster Number")
plt.show()

# Finding optimal number of clusters using silhouette score
silhouette_scores = []
for i in range(2, 11):
    labels = evaluate_kmeans(X_scaled, i)[0]
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal Cluster Number")
plt.show()

print("Optimal number of clusters:", optimal_clusters)

# Clustering using KMeans on PCA-transformed data
pca_clusters, _ = evaluate_kmeans(X_pca, optimal_clusters)

# Clustering using KMeans on LDA-transformed data
lda_clusters, _ = evaluate_kmeans(X_lda, optimal_clusters)

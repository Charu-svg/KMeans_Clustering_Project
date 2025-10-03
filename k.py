# Run this once in a terminal or notebook cell
# In Colab: prefix with ! e.g. !pip install ...


# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Optional: show plots inline in notebooks

# Put Mall_Customers.csv in the same folder as the notebook, or give the path here
path = "Mall_Customers.csv"

if os.path.exists(path):
    df = pd.read_csv(path)
    print("Loaded", path)
else:
    # If you don't have the file yet, create a small synthetic example to practice:
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=220, centers=4, cluster_std=1.8, random_state=42)
    df = pd.DataFrame(X, columns=["Annual Income (k$)","Spending Score (1-100)"])
    df["Age"] = np.random.randint(18,70,size=len(df))
    df.insert(0, "CustomerID", range(1, len(df)+1))
    print("Using synthetic dataset (no CSV found).")

df.head()
print(df.shape)
print(df.describe())

# Scatter of the two features commonly used for mall segmentation:
plt.figure(figsize=(6,4))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Income vs Spending Score (raw)")
plt.show()
# Choose features to cluster on:
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values

# Scale (important for KMeans!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,4))
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title("PCA (2D) projection of scaled features")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()
# Elbow: inertia for k=1..10
inertias = []
K = range(1,11)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertias, marker='o')
plt.title("Elbow Method (Inertia vs k)")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.xticks(K)
plt.show()

# Silhouette: for k=2..10 (silhouette undefined for k=1)
sil_scores = []
K2 = range(2,11)
for k in K2:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(6,4))
plt.plot(list(K2), sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k"); plt.ylabel("Silhouette Score")
plt.xticks(list(K2))
plt.show()

# Choose a k: you can pick from elbow visual or the k with highest silhouette.
best_k_by_sil = list(K2)[int(np.argmax(sil_scores))]
print("Suggested k by highest silhouette:", best_k_by_sil)
# Replace best_k with the k you choose (from above)
best_k = best_k_by_sil

kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df["Cluster"] = labels

# PCA scatter with clusters
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)  # centroids in PCA coordinates

plt.figure(figsize=(6,4))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)   # let matplotlib pick default colors
plt.scatter(centroids_pca[:,0], centroids_pca[:,1], marker='X', s=140)
plt.title(f"KMeans clusters (k={best_k}) in PCA space")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

print("Silhouette score for final clustering:", round(silhouette_score(X_scaled, labels),4))
df.head(12)
best_k = 5  # Example: chosen from elbow/silhouette
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels
print(df.head())
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
plt.scatter(centroids_pca[:,0], centroids_pca[:,1], marker="X", s=200, color="red")
plt.title(f"K-Means Clusters (k={best_k})")
plt.show()
df.to_csv("clustered_results.csv", index=False)

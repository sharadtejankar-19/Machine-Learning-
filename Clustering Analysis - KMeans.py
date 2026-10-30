import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

data = load_iris()
X = data.data

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print("Optimal Number of Clusters:", 3)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

print("\nNumber of Data Points in Each Cluster:")

for i in range(3):
    print("Cluster", i, ":", sum(clusters == i))

print("\nFirst 20 Cluster Labels:")
print(clusters[:20])

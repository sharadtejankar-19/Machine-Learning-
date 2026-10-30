import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model without PCA
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
pred1 = model.predict(X_test)

acc1 = accuracy_score(y_test, pred1)

# Apply PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Model with PCA
model = LogisticRegression(max_iter=5000)
model.fit(X_train_pca, y_train)
pred2 = model.predict(X_test_pca)

acc2 = accuracy_score(y_test, pred2)

# Display results
print("Original Features:", X.shape[1])
print("Reduced Features:", X_train_pca.shape[1])
print("Accuracy without PCA:", acc1)
print("Accuracy with PCA:", acc2)

# Plot accuracy
plt.bar(["Without PCA", "With PCA"], [acc1, acc2])
plt.ylabel("Accuracy")
plt.title("PCA Performance Comparison")
plt.show()
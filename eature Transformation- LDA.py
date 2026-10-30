import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Model without LDA
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
pred1 = model.predict(X_test)

acc1 = accuracy_score(y_test, pred1)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Model with LDA
model = LogisticRegression(max_iter=5000)
model.fit(X_train_lda, y_train)
pred2 = model.predict(X_test_lda)

acc2 = accuracy_score(y_test, pred2)

# Display results
print("Original Features:", X.shape[1])
print("Reduced Features:", X_train_lda.shape[1])
print("Accuracy without LDA:", acc1)
print("Accuracy with LDA:", acc2)

# Plot accuracy
plt.figure(figsize = (4,3))
plt.bar(["Without LDA", "With LDA"], [acc1, acc2],width=0.1)
plt.ylabel("Accuracy")
plt.title("LDA Performance Comparison")
plt.show()

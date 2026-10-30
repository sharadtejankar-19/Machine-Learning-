
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model1 = LogisticRegression(max_iter=5000)
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

pred2 = random_forest.predict(X_test)

print("Model1: Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, pred1), 4))
print("Precision:", round(precision_score(y_test, pred1), 4))
print("Recall:", round(recall_score(y_test, pred1), 4))
print("F1-Score:", round(f1_score(y_test, pred1), 4))

print("\nRandom Forest Classifier")
print("Accuracy:", round(accuracy_score(y_test, pred2), 4))
print("Precision:", round(precision_score(y_test, pred2), 4))
print("Recall:", round(recall_score(y_test, pred2), 4))
print("F1-Score:", round(f1_score(y_test, pred2), 4))

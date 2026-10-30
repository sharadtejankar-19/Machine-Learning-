
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
data = pd.read_csv("uber.csv")

# Pre-processing
data = data.dropna()

data = data[
    (data["fare_amount"] > 0) &
    (data["passenger_count"] > 0) &
    (data["pickup_longitude"].between(-180, 180)) &
    (data["dropoff_longitude"].between(-180, 180)) &
    (data["pickup_latitude"].between(-90, 90)) &
    (data["dropoff_latitude"].between(-90, 90))
].copy()

# Calculate distance using coordinates
lat1 = np.radians(data["pickup_latitude"])
lat2 = np.radians(data["dropoff_latitude"])
lon1 = np.radians(data["pickup_longitude"])
lon2 = np.radians(data["dropoff_longitude"])

a = (np.sin((lat2-lat1)/2)**2 +
     np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)

data["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))

# Extract time features
data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
data = data.dropna(subset=["pickup_datetime"])

data["hour"] = data["pickup_datetime"].dt.hour
data["day"] = data["pickup_datetime"].dt.day
data["month"] = data["pickup_datetime"].dt.month

# Remove zero-distance and extreme-distance rides
data = data[(data["distance"] > 0) & (data["distance"] < 100)]

# Record outliers using IQR
Q1 = data["fare_amount"].quantile(0.25)
Q3 = data["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

outliers = data[
    (data["fare_amount"] < Q1 - 1.5 * IQR) |
    (data["fare_amount"] > Q3 + 1.5 * IQR)
]

print("Number of fare outliers:", len(outliers))

# Remove fare outliers
data = data[
    (data["fare_amount"] >= Q1 - 1.5 * IQR) &
    (data["fare_amount"] <= Q3 + 1.5 * IQR)
].copy()

# Check correlation
features = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "distance", "hour", "day", "month",
    "fare_amount"
]

plt.figure(figsize=(8, 6))
sns.heatmap(data[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Select features and target
X = data[features[:-1]]
y = data["fare_amount"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

# Train and compare
results = {}

for name, model in models.items():
    model = make_pipeline(StandardScaler(), model)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)

    results[name] = [r2, rmse, mae]

    print("\n", name)
    print("R2 Score:", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)

# Compare models
result_df = pd.DataFrame(
    results, index=["R2 Score", "RMSE", "MAE"]
).T

print("\nModel Comparison:")
print(result_df)

# Plot R2 scores
result_df["R2 Score"].plot(
    kind="bar", figsize=(5, 3), title="Model Comparison"
)
plt.ylabel("R2 Score")
plt.tight_layout()
plt.show()

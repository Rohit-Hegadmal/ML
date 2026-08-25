import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("Health.csv")

# Convert 'sex' to numbers
data["sex"] = data["sex"].map({"Boy": 0, "Girl": 1})

# Features and target
X = data.drop("bdi_total", axis=1)
y = data["bdi_total"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print predictions
print("Predicted Values:")
print(predictions)

# Model score
print("Accuracy (R² Score):", model.score(X_test, y_test))
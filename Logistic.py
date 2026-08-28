import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,confusion_matrix,f1_score

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

data = data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()

# Convert 'Sex' to numbers
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print predictions
print("Predicted Values:")
print(predictions)

# Model score
print("Accuracy Score:", model.score(X_test, y_test))
print("F1 Score:", f1_score(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Load the Cancer Dataset
data = pd.read_csv("Cancer_Data.csv")

# Display first 5 records
print("First 5 records:")
print(data.head())

# Display dataset shape
print("\nData Shape:", data.shape)

# Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# Convert diagnosis into numerical values
# M = 1 (Malignant)
# B = 0 (Benign)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Separate features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Split dataset into training set (70%) and testing set (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=109
)

# Create SVM Classifier using Linear Kernel
clf = svm.SVC(kernel="linear")

# Train the SVM model
clf.fit(X_train, y_train)

# Predict the test dataset
y_pred = clf.predict(X_test)

# Model Evaluation
print("\n--- SVM Model Evaluation ---")

# Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Precision
print("Precision:", metrics.precision_score(y_test, y_pred))

# Recall
print("Recall:", metrics.recall_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred))
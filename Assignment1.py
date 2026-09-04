# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
data = pd.read_csv("iris.csv")

# First 5 rows
print("First 5 rows:")
print(data.head())

# Dataset information
print("\nShape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nInformation:")
data.info()

# Statistics
print("\nStatistics:")
print(data.describe())

# Missing values
print("\nMissing values:")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

# Check duplicates
print("\nDuplicate rows:")
print(data.duplicated().sum())

# Remove duplicates
data = data.drop_duplicates()

# Species count
print("\nSpecies count:")
print(data["species"].value_counts())


# Species graph
sns.countplot(data=data, x="species")
plt.title("Species Count")
plt.show()


# Histograms
data.hist(figsize=(10, 7))
plt.tight_layout()
plt.show()


# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data.iloc[:, 0:4])
plt.title("Boxplot")
plt.show()


# Outliers using Z-score
z = np.abs(stats.zscore(data.iloc[:, 0:4]))

print("\nOutliers:")
print((z > 3).sum())


# Correlation
print("\nCorrelation:")
print(data.iloc[:, 0:4].corr())


# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    data.iloc[:, 0:4].corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.show()


# Scatter plot
sns.scatterplot(
    data=data,
    x="sepal_length",
    y="sepal_width",
    hue="species"
)
plt.title("Sepal Length vs Sepal Width")
plt.show()


# Petal scatter plot
sns.scatterplot(
    data=data,
    x="petal_length",
    y="petal_width",
    hue="species"
)
plt.title("Petal Length vs Petal Width")
plt.show()


# Pairplot
sns.pairplot(
    data,
    vars=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ],
    hue="species"
)
plt.show()


# Mean of each species
print("\nSpecies-wise Mean:")
print(
    data.groupby("species").mean(numeric_only=True)
)
# K-Nearest Neighbors (KNN) Classifier on Breast Cancer Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
cancer = load_breast_cancer()

# Convert to DataFrame for easier handling
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

print("Dataset shape:", df.shape)
print("\nTarget Names:", cancer.target_names)
print("\nFeature Columns:\n", df.columns[:10].tolist(), "...")

# Split Data into Features and Target
X = df.drop('target', axis=1)
y = df['target']

# Standardize Features (very important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Find the Best k using Cross-Validation
k_values = list(range(1, 21, 2))
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest k found: {best_k} with CV accuracy = {max(cv_scores):.4f}")

# Train KNN Model with Best k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Predict on Test Data
y_pred = knn.predict(X_test)

# Evaluate Model
print("\n Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='coolwarm', interpolation='nearest')
plt.title('Confusion Matrix — KNN on Breast Cancer Dataset')
plt.colorbar()
plt.xticks([0, 1], cancer.target_names)
plt.yticks([0, 1], cancer.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white', fontsize=12)
plt.show()

# Accuracy vs K plot
plt.figure(figsize=(8,5))
plt.plot(k_values, cv_scores, marker='o')
plt.title('K vs Cross-Validation Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('CV Accuracy')
plt.grid(True)
plt.show()

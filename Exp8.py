# Support Vector Machine (SVM) Classifier on Housing Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("Housing.csv")

# 2. Convert price into categories (Low, Medium, High)
df['price_category'] = pd.cut(df['price'], bins=3, labels=['Low', 'Medium', 'High'])

# 3. Select Features and Target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price_category']

# 4. Scale features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train SVM Classifier (RBF kernel)
svm_clf = SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
svm_clf.fit(X_train, y_train)

# 7. Make Predictions
y_pred = svm_clf.predict(X_test)

# 8. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix — SVM Classifier (Housing Dataset)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

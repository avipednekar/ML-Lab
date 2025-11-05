# Exp7: Random Forest Classifier on Housing Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("Housing.csv")

# 2. Convert price into balanced categories using qcut
df['price_category'] = pd.qcut(df['price'], q=3, labels=['Low', 'Medium', 'High'])

# Show how many samples per class
print("Class distribution:")
print(df['price_category'].value_counts(), "\n")

# 3. Select Features and Target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price_category']

# 4. Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split Data (Stratify ensures class balance in train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=5,
    random_state=42
)
rf_classifier.fit(X_train, y_train)

# 7. Make Predictions
y_pred = rf_classifier.predict(X_test)

# 8. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")

# Classification report (with zero_division=0 to avoid warnings)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix — Random Forest (Housing Dataset)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

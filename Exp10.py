#  Naive Bayes Classifier on Email Message Dataset (Spam Detection)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'message': [
        'Win money now!!! Click here to claim your prize',
        'Limited time offer, buy 1 get 1 free',
        'Hey, can we meet for lunch tomorrow?',
        'Your loan has been approved, contact immediately',
        'Important project meeting scheduled for today',
        'Congratulations! You have won a free iPhone',
        'Please review the attached report',
        'You are selected for a cash reward',
        'Let’s go out for dinner tonight',
        'Get cheap medicines online, discount 50%'
    ],
    'label': [
        'spam', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam'
    ]
}

df = pd.DataFrame(data)
print(" Dataset Preview:\n", df.head(), "\n")

# Split Data into Features (X) and Target (y)
X = df['message']
y = df['label']

#  Split into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Convert Text to Numeric Features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on Test Data
y_pred = model.predict(X_test_tfidf)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f" Model Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix — Naive Bayes Email Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Test with a New Email
new_email = ["Congratulations, you won a trip to Paris!"]
new_email_tfidf = vectorizer.transform(new_email)
prediction = model.predict(new_email_tfidf)
print(f" New Email: '{new_email[0]}'")
print(f"Predicted Category: {prediction[0].upper()}")

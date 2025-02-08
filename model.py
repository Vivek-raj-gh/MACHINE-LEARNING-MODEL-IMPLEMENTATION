import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame({
    'text': [
        'Win a free iPhone now!',
        'Hello, how are you?',
        'Free money waiting for you!',
        'Meeting at 10 AM tomorrow.',
        'Get cash back on every purchase!',
        'This is a normal email.',
        'Claim your lottery prize now!',
        'Can we schedule the appointment?'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam
})

print("Data preview:")
print(data.head())
print("\nData statistics:")
print(data['label'].value_counts())

sns.countplot(data['label'], palette='viridis')
plt.title("Spam vs Not Spam Email Count")
plt.show()

X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix for Spam Detection")
plt.show()

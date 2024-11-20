import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample data
data = {
    'review': [
        "I loved this movie, it was fantastic!",
        "Terrible movie, I hated it.",
        "Amazing story and great acting.",
        "Not worth the time, boring and slow.",
        "It was okay, not great but not terrible either.",
    ],
    'sentiment': [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# Step 1: Data Preprocessing
def clean_text(text):
    """Clean and preprocess the text."""
    text = re.sub(r'\W', ' ', text) 
    text = text.lower().strip()  
    return text

df['cleaned_review'] = df['review'].apply(clean_text)


stop_words = set(stopwords.words('english'))
df['cleaned_review'] = df['cleaned_review'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in stop_words)
)

# Step 2: Feature Engineering (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review']).toarray()
y = df['sentiment']

# Step 3: Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = LogisticRegression(max_iter=200)  # Increase max_iter for convergence
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 6: Testing with New Data
new_reviews = [
    "The movie was absolutely wonderful!",
    "I didn't like the movie at all, it was a waste of time."
]
new_reviews_cleaned = [clean_text(review) for review in new_reviews]
new_reviews_cleaned = [' '.join(word for word in review.split() if word not in stop_words)
                       for review in new_reviews_cleaned]
new_X = vectorizer.transform(new_reviews_cleaned).toarray()
predictions = model.predict(new_X)

for review, sentiment in zip(new_reviews, predictions):
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"Review: {review}\nSentiment: {sentiment_label}\n")


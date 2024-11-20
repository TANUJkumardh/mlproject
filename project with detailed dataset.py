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

# Step 1: Load Dataset from NLTK
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews


reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]


data = pd.DataFrame(reviews, columns=["review", "sentiment"])


data['review'] = data['review'].apply(lambda x: ' '.join(x))

# Step 2: Data Preprocessing
def clean_text(text):
    """Clean and preprocess the text."""
    text = re.sub(r'\W', ' ', text) 
    text = text.lower().strip()  
    return text

data['cleaned_review'] = data['review'].apply(clean_text)


data['sentiment'] = data['sentiment'].map({'pos': 1, 'neg': 0})

# Step 3: Remove Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
data['cleaned_review'] = data['cleaned_review'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in stop_words)
)

# Step 4: Feature Engineering (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment']

# Step 5: Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LogisticRegression(max_iter=200)  
model.fit(X_train, y_train)

# Step 7: Model Evaluation
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

# Step 8: Testing with New Data
new_reviews = [
    "The movie was an absolute masterpiece!",
    "I regret wasting my time watching that film."
]
new_reviews_cleaned = [clean_text(review) for review in new_reviews]
new_reviews_cleaned = [' '.join(word for word in review.split() if word not in stop_words)
                       for review in new_reviews_cleaned]
new_X = vectorizer.transform(new_reviews_cleaned).toarray()
predictions = model.predict(new_X)

for review, sentiment in zip(new_reviews, predictions):
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"Review: {review}\nSentiment: {sentiment_label}\n")

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Create visuals folder if it doesn’t exist
os.makedirs("visuals", exist_ok=True)

# Load dataset (update path if needed)
data = pd.read_csv("dataset/IMDB Dataset.csv")

# Preprocessing function
def clean_text(text):
    text = re.sub('<.*?>', '', text)  # remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # keep only letters
    text = text.lower()  # lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

data['review'] = data['review'].apply(clean_text)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42
)

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train SVM classifier
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Sentiment Analysis")
plt.savefig("visuals/confusion_matrix.png")
plt.close()

# WordCloud for Positive and Negative reviews
positive_text = " ".join(data[data['sentiment'] == 1]['review'])
negative_text = " ".join(data[data['sentiment'] == 0]['review'])

plt.figure(figsize=(12, 6))

# Positive
plt.subplot(1, 2, 1)
wc = WordCloud(width=400, height=400, background_color="white").generate(positive_text)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Positive Reviews")

# Negative
plt.subplot(1, 2, 2)
wc = WordCloud(width=400, height=400, background_color="black", colormap="Reds").generate(negative_text)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Negative Reviews")

plt.savefig("visuals/wordclouds.png")
plt.close()

# Test with custom review
new_review = "The movie was absolutely wonderful, full of emotions!"
new_review_clean = clean_text(new_review)
new_review_vec = vectorizer.transform([new_review_clean])
prediction = model.predict(new_review_vec)[0]

print("\nPrediction for new review:", "Positive" if prediction == 1 else "Negative")
print("Visualizations saved in the 'visuals/' folder")

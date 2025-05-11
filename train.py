import pandas as pd
import json
import pickle
import re
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC  # More accurate than Naïve Bayes
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
with open("Dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract text and labels
texts = [item["content"] for item in data]
labels = [int(item["annotation"]["label"][0]) for item in data]

# Convert to DataFrame
df = pd.DataFrame({"text": texts, "label": labels})

# Print original label distribution
print("Original Label Distribution:", Counter(labels))

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return " ".join(words)

# Apply text preprocessing
df["text"] = df["text"].apply(preprocess_text)

# Balance dataset using Oversampling
bullying = df[df["label"] == 1]
not_bullying = df[df["label"] == 0]

if len(bullying) > len(not_bullying):
    not_bullying = resample(not_bullying, replace=True, n_samples=len(bullying), random_state=42)  
else:
    bullying = resample(bullying, replace=True, n_samples=len(not_bullying), random_state=42)  

df_balanced = pd.concat([bullying, not_bullying]).sample(frac=1, random_state=42)

# Print balanced label distribution
print("Balanced Label Distribution:", Counter(df_balanced['label']))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df_balanced["text"], df_balanced["label"], test_size=0.2, random_state=42)

# Feature Extraction (TF-IDF with n-grams)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model (Using SVM for Higher Accuracy)
model = SVC(kernel="linear", probability=True)  # More accurate than Naïve Bayes
model.fit(X_train_vec, y_train)

# Evaluate Model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Test model on sample texts
test_sentences = ["You are so dumb!", "Hope you have a nice day!", "You are a loser!", "I love coding in Python!"]
test_vec = vectorizer.transform(test_sentences)
test_preds = model.predict(test_vec)

for text, pred in zip(test_sentences, test_preds):
    print(f"Text: {text} --> Prediction: {'Bullying' if pred == 1 else 'Not Bullying'}")

# Save model and vectorizer
with open("cyberbully_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model training completed with higher accuracy!")

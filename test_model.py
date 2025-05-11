import pickle

# Load trained model and vectorizer
model = pickle.load(open("cyberbully_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sample test sentences
test_sentences = [
    "I hate you, you are so dumb!",   
    "Have a nice day, friend!",       
    "You're a loser, go away!",       
    "I love coding in Python!",       
]

# Predict on sample texts
for text in test_sentences:
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    print(f"Text: {text} --> Prediction: {'Bullying' if prediction == 1 else 'Not Bullying'}")

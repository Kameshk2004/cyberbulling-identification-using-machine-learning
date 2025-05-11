import pickle
from imgToText import extract_text_from_image
from gemini_ai import get_gemini_response

# Load trained model and vectorizer
model = pickle.load(open("cyberbully_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess_input(text):
    return text.strip().lower()

def predict_text(input_text):
    """Predicts if the input text is bullying or not, cross-checks with Gemini AI, and rephrases if needed."""
    if not input_text.strip():
        return "Invalid input", None

    input_text = preprocess_input(input_text)
    text_vec = vectorizer.transform([input_text])
    prediction = model.predict(text_vec)[0]
    result = "Bullying" if prediction == 1 else "Not Bullying"

    try:
        ai = get_gemini_response(f"Give this statement 1 or 0 respectively by bullying or not bullying respectively (just return 1 or 0): {input_text}")
        ai = int(str(ai).strip())
    except:
        ai = prediction  # fallback to model prediction if AI fails

    if prediction == 1 or ai == 1:
        try:
            ai_rephrase = get_gemini_response(f"Rephrase this sentence to be polite and non-bullying. Give me just the most suitable one sentence: {input_text}")
            return result, ai_rephrase
        except:
            return result, None
    else:
        return result, None

def main():
    image_path = input("\nEnter image path (or press Enter to input text manually): ").strip()

    if image_path:
        extracted_text = extract_text_from_image(image_path)
        if extracted_text:
            print("\nExtracted Text:", extracted_text)
            result, ai_rephrase = predict_text(extracted_text)
            print("\nPrediction:", result)
            if ai_rephrase:
                print("\nSuggested Non-Bullying Version:", ai_rephrase)
        else:
            print("No text found in the image!")
    else:
        while True:
            text = input("\nEnter a sentence (or type 'exit' to stop): ").strip()
            if text.lower() == "exit":
                print("Exiting...")
                break

            result, ai_rephrase = predict_text(text)
            print("\nPrediction:", result)
            if ai_rephrase:
                print("\nSuggested Non-Bullying Version:", ai_rephrase)

if __name__ == "__main__":
    main()

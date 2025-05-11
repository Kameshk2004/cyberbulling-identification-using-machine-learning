import google.generativeai as genai

API_KEY = "AIzaSyA84eJc8ZxiogpEA0IPiWM__Un9vaJ64Fw"
genai.configure(api_key=API_KEY)


def get_gemini_response(prompt):
    
    model = genai.GenerativeModel("gemini-1.5-pro")  
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    user_input = input("Ask AI: ")
    response = get_gemini_response(user_input)
    print("\nGemini AI Response:\n", response)

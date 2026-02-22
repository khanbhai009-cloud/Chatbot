import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) so your frontend can talk to this API
CORS(app)

# ==========================================
# ⚙️ CONFIGURATION: API KEY AND MODELS
# ==========================================
# The API key is securely loaded from the .env file. Never hardcode it here!
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize the OpenAI client pointing to OpenRouter's URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 🚀 MULTIPLE AI FALLBACK LIST (Priority Order)
# The system will try these models one by one from top to bottom.
AI_MODELS = [
    "z-ai/glm-4.5-air:free",                  # Priority 1 (Primary Model)
    "deepseek/deepseek-r1-0528:free",         # Priority 2 (First Backup)
    "meta-llama/llama-3.3-70b-instruct:free"  # Priority 3 (Final Backup)
]


# ==========================================
# 🧠 CHATBOT PERSONALITY (SYSTEM PROMPT)
# ==========================================
# This is the "soul" of your chatbot. It tells the AI how to behave.
SYSTEM_PROMPT = """
You are a highly capable, warm, and premium personal assistant. 
Your goal is to make the user's life easier by providing clear, helpful, and insightful answers.

Key personality traits:
- Tone: Conversational, polite, deeply empathetic, and human-like. Avoid robotic phrases like "As an AI...".
- Style: Well-structured and easy to read. Use bullet points, bold text, and paragraphs appropriately.
- Vibe: Imagine you are a high-end concierge or a trusted chief of staff. You are smart, professional, but very approachable.

Always consider the conversation history provided to you to give context-aware responses.
"""

@app.route('/chat', methods=['POST'])
def chat():
    """
    POST /chat
    Expected JSON Payload:
    {
      "message": "Hello!",
      "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    }
    """
    try:
        # 1. Parse incoming JSON data
        data = request.get_json()

        # 2. Validate the input
        if not data or not data.get("message"):
            return jsonify({
                "status": "error",
                "message": "Validation Error: 'message' field is required and cannot be empty."
            }), 400

        user_message = data.get("message").strip()
        # Get history if it exists, otherwise use an empty list
        chat_history = data.get("history", []) 

        # 3. Construct the message array for the AI
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        # ==========================================
        # 🔄 THE FALLBACK LOGIC ENGINE
        # ==========================================
        ai_reply = None
        
        # Loop through our list of models
        for model in AI_MODELS:
            try:
                print(f"🔄 Attempting to generate response using: {model}...")
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                
                # If successful, save the reply and break out of the loop
                ai_reply = response.choices[0].message.content
                print(f"✅ Success with {model}!")
                break  
            
            except Exception as model_error:
                # If this model fails, print the error and continue to the next one
                print(f"⚠️ {model} failed. Error: {str(model_error)}")
                continue  

        # 5. Final check: Did ALL models fail?
        if not ai_reply:
            print("❌ CRITICAL: All fallback models failed.")
            return jsonify({
                "status": "error",
                "message": "Sorry, all our AI servers are currently overloaded. Please try again in a few seconds."
            }), 500

        # 6. Send successful response back to the frontend
        return jsonify({
            "status": "success",
            "reply": ai_reply
        }), 200

    except Exception as e:
        # Catch any server/code errors and return a clean JSON error
        print(f"Error during chat processing: {str(e)}") 
        return jsonify({
            "status": "error",
            "message": "Oops! Something went wrong on the server side. Please try again later."
        }), 500

# Run the app locally
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

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
# ⚙️ CONFIGURATION: API KEY AND MODEL
# ==========================================
# The API key is securely loaded from the .env file. Never hardcode it here!
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize the OpenAI client pointing to OpenRouter's URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# You can easily change the model here in the future
AI_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

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
        # Start with the System Prompt, append the history, then add the new user message
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        # 4. Make the API call to OpenRouter
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
        )

        # 5. Extract the AI's reply
        ai_reply = response.choices[0].message.content

        # 6. Send successful response back to the frontend
        return jsonify({
            "status": "success",
            "reply": ai_reply
        }), 200

    except Exception as e:
        # Catch any errors (e.g., API downtime, network issues) and return a clean JSON error
        print(f"Error during chat processing: {str(e)}") # Log for server terminal
        return jsonify({
            "status": "error",
            "message": "Oops! Something went wrong while connecting to the AI. Please try again later."
        }), 500

# Run the app locally (this is ignored by Gunicorn in production)
if __name__ == '__main__':
    # Debug=True is great for local development, automatically reloads on code changes
    app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, render_template, request, jsonify
#from chatBot import get_diamond_recommendations  # Import your function

app = Flask(__name__)

messages = [
    {"text": "Hello! I am your diamond assistant.", "user": False, "avatar": "/static/bot_avatar.png"}
]
style = ""

@app.route("/")
def index():
    return render_template("index.html", messages=messages)

@app.route("/api/chat", methods=["POST"])
def chat():
    global style
    user_input = request.json["query"]
    messages.append({"text": user_input, "user": True, "avatar": "/static/user_avatar.png"})

    if not style and user_input.lower() not in ['hi', 'hello']:
        messages.append({"text": "Please specify the style (LabGrown or Natural):", "user": False, "avatar": "/static/bot_avatar.png"})
        style = "pending"
        return jsonify({"response": ""})

    if style == "pending":
        normalized_input = user_input.lower()
        if normalized_input in ['labgrown', 'natural']:
            style = normalized_input
            messages.append({"text": f"Great! What are you looking for in your {style} diamond?", "user": False, "avatar": "/static/bot_avatar.png"})
        else:
            messages.append({"text": "Please specify either 'LabGrown' or 'Natural'.", "user": False, "avatar": "/static/bot_avatar.png"})
        return jsonify({"response": ""})

    # --- Call your chatbot function (NO CHANGES TO THE RESPONSE TEXT) ---
    response_text = get_diamond_recommendations(user_input, style)

    messages.append({"text": response_text, "user": False, "avatar": "/static/bot_avatar.png"})
    return jsonify({"response": response_text})  # Return the *exact* response

if __name__ == "__main__":
    app.run(debug=True)
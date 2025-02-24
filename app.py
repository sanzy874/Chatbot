from flask import Flask, render_template, request, jsonify
import io
import sys
from chatBot import diamond_chatbot, load_data_and_index
from groq import Groq
from dotenv import load_dotenv
import os

app = Flask(__name__)

# File paths
EMBEDDING_FILE_PATH = 'diamond_embeddings.npy'
FAISS_INDEX_FILE = 'diamond_faiss_index.faiss'
DATAFRAME_FILE = 'diamond_dataframe.csv'
MODEL_PATH = 'sentence_transformer_model'

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq()

# Load data, embeddings, FAISS index, and model at startup
try:
    df, embeddings, faiss_index, model = load_data_and_index(

        EMBEDDING_FILE_PATH,
        FAISS_INDEX_FILE,
        DATAFRAME_FILE,
        MODEL_PATH
    )
    print("Successfully loaded diamond data and models")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()

        if not user_query:
            return jsonify({
                'response': "I'm your diamond assistant. How can I help you find the perfect diamond today?"
            })

        # Capture printed output from diamond_chatbot
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        # Call the chatbot logic
        diamond_chatbot(user_query, df, faiss_index, model, client)

        # Restore stdout and get response
        sys.stdout = old_stdout
        response = mystdout.getvalue().strip()

        # If no response was generated, provide a default
        if not response:
            response = "I'm having trouble understanding your request. Could you please provide more details about the diamond you're looking for?"

        return jsonify({'response': response})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'response': "I apologize, but I encountered an error. Please try your request again."
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
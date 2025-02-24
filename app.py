from flask import Flask, render_template, request
import io
import sys
import os
from chatBot import diamond_chatbot, load_data_and_index  # Import your backend logic
from groq import Groq  # If your chatbot uses Groq for generating responses

app = Flask(__name__)

# File paths (adjust as needed)
EMBEDDING_FILE_PATH = 'diamond_embeddings.npy'
FAISS_INDEX_FILE = 'diamond_faiss_index.faiss'
DATAFRAME_FILE = 'diamond_dataframe.csv'
MODEL_PATH = 'sentence_transformer_model'
FILE_PATH = 'diamonds.csv'  # if your chatBot.py uses this for first-time creation

# Load data, embeddings, FAISS index, and model once at startup
df, embeddings, faiss_index, model = load_data_and_index(EMBEDDING_FILE_PATH, FAISS_INDEX_FILE, DATAFRAME_FILE, MODEL_PATH)

# Set the GROQ_API_KEY environment variable
os.environ["GROQ_API_KEY"] = "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"  

# Initialize your Groq client (if required by your chatbot logic)
client = Groq()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.form.get('user_query')
    # Capture printed output from diamond_chatbot()
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    
    # Call the chatbot logic from chatBot.py
    diamond_chatbot(user_query, df, faiss_index, model, client)
    
    sys.stdout = old_stdout
    response = mystdout.getvalue()
    return response

if __name__ == '__main__':
    app.run(debug=False)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
from groq import Groq
from dotenv import load_dotenv
# Global: Path for embeddings (used in temporary index building)
EMBEDDING_FILE_PATH = 'diamond_embeddings.npy'

# def setup_chatbot():
#     # 1. Load or create df, embeddings, index, model
#     df, embeddings, index, model = load_data_and_index(
#         EMBEDDING_FILE_PATH, 
#         FAISS_INDEX_FILE, 
#         DATAFRAME_FILE, 
#         MODEL_PATH
#     )
#     # 2. Initialize Groq or any other client
#     client = Groq()

#     return df, index, model, client


# ------------------- Data Preparation & Embedding Generation -------------------
def data_and_embedding(file_path, embedding_file, faiss_index_file, dataframe_file, model_path):
    df = pd.read_csv(file_path)
    df = df.replace({r'[^\x00-\x7F]+': ''}, regex=True)
    # Convert all data values to lowercase
    df = df.apply(lambda x: x.astype(str).str.lower())

    print(f"Number of rows in dataset: {df.shape[0]}")

    # Create a combined text field that includes Style
    df['combined_text'] = (
        "Style: " + df['Style'].astype(str) + ", " +
        "Carat: " + df['Carat'].astype(str) + ", " +
        "Clarity: " + df['Clarity'].astype(str) + ", " +
        "Color: " + df['Color'].astype(str) + ", " +
        "Cut: " + df['Cut'].astype(str) + ", " +
        "Shape: " + df['Shape'].astype(str) + ", " +
        "Price: " + df['Price'].astype(str) + ", " +
        "Lab: " + df['Lab'].astype(str) + ", " +
        "Polish: " + df['Polish'].astype(str) + ", " +
        "Symmetry: " + df['Symmetry'].astype(str)
    )

    # Ensure Carat is numeric
    df["Carat"] = pd.to_numeric(df["Carat"], errors="coerce")

    print("First combined text:", df['combined_text'].iloc[0])

    # Generate embeddings using SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(df['combined_text'].tolist(), convert_to_numpy=True)
    print(f"Shape of embeddings: {embeddings.shape}")

    # Build FAISS index using L2 distance
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings)

    # Save embeddings, FAISS index, and dataframe to disk
    np.save(embedding_file, embeddings)
    faiss.write_index(index, faiss_index_file)
    df.to_csv(dataframe_file, index=False)
    model.save(model_path)

    print("Model, embeddings, and FAISS index saved to disk.")
    return df, embeddings, index, model

# ------------------- Load Data & FAISS Index -------------------
def load_data_and_index(embedding_file, faiss_index_file, dataframe_file, model_path):
    df = pd.read_csv(dataframe_file)
    df["Carat"] = pd.to_numeric(df["Carat"], errors="coerce")
    embeddings = np.load(embedding_file)
    index = faiss.read_index(faiss_index_file)
    model = SentenceTransformer(model_path)
    print("Loaded data, embeddings, FAISS index, and model from disk.")
    return df, embeddings, index, model

# ------------------- Utility: Extract Constraints from Query -------------------
def extract_constraints_from_query(user_query):
    """
    Extracts constraints (Carat, Color, Clarity, Cut, Symmetry, Polish, Style, Shape)
    from the user's query. Non-numeric values are normalized to lowercase.
    Returns a dictionary.
    """
    constraints = {}

    # Extract Carat (e.g., "0.8 carat")
    carat_match = re.search(r'(\d+(\.\d+)?)\s*-?\s*carat', user_query, re.IGNORECASE)
    if carat_match:
        constraints["Carat"] = float(carat_match.group(1))

    # Extract "under price 2000" or "under 2000"
    budget_match = re.search(r'under(?:\s*price)?\s*(\d+)', user_query, re.IGNORECASE)
    if budget_match:
        constraints["Budget"] = float(budget_match.group(1))

    # Extract Color (e.g., "E", "G") – normalized to lowercase
    color_match = re.search(r'\b([a-j])\b', user_query, re.IGNORECASE)
    if color_match:
        constraints["Color"] = color_match.group(1).lower()

    # Extract Clarity (e.g., "VS1", "VVS2", etc.) – normalized to lowercase
    clarity_match = re.search(r'\b(if|vvs1|vvs2|vs1|vs2|si1|si2)\b', user_query, re.IGNORECASE)
    if clarity_match:
        constraints["Clarity"] = clarity_match.group(1).lower()

    # Extract Cut (e.g., "excellent", "ideal", "very good", "good")
    cut_match = re.search(r'\b(excellent|ideal|very good|good)\b', user_query, re.IGNORECASE)
    if cut_match:
        constraints["Cut"] = cut_match.group(1).lower()

    # Extract Symmetry (e.g., "excellent", "very good", "good")
    symmetry_match = re.search(r'\b(excellent|very good|good)\b', user_query, re.IGNORECASE)
    if symmetry_match:
        constraints["Symmetry"] = symmetry_match.group(1).lower()

    # Extract Polish (e.g., "excellent", "ideal", "very good", "good")
    polish_match = re.search(r'\b(excellent|ideal|very good|good)\b', user_query, re.IGNORECASE)
    if polish_match:
        constraints["Polish"] = polish_match.group(1).lower()

    # Extract Style (e.g., "labgrown" or "natural") – normalized to lowercase
    style_match = re.search(r'\b(labgrown|natural)\b', user_query, re.IGNORECASE)
    if style_match:
        constraints["Style"] = style_match.group(1).lower()

    # Extract Shape (e.g., "round", "princess", "emerald", etc.) – normalized to lowercase
    shape_match = re.search(r'\b(round|princess|emerald|asscher|cushion|marquise|radiant|oval|pear|heart|square radiant)\b', user_query, re.IGNORECASE)
    if shape_match:
        constraints["Shape"] = shape_match.group(1).lower()

    return constraints

# ------------------- Hybrid Search (Semantic + Filter + Composite Ranking) -------------------
def hybrid_search(user_query, df, faiss_index, model, top_k=200):
    """
    1. Extract constraints from the query.
    2. If Style is specified, restrict the DataFrame to that style.
    3. If Carat is specified, pre-filter for near-exact matches using a narrow tolerance.
    4. Perform FAISS search on the (possibly pre-filtered) dataset.
    5. Compute a composite score that prioritizes:
        - Exact Carat match (highest weight)
        - Then Price (lower is better)
        - Then mismatches in Clarity and Color (penalties)
        - Then mismatches in Cut, Symmetry, and Polish (lower penalty)
    6. Return the top 5 results.
    """
    constraints = extract_constraints_from_query(user_query)

    # Restrict by Style if specified
    if "Style" in constraints:
        df = df[df['Style'] == constraints["Style"]]
        if df.empty:
            print("No diamonds found for the specified style.")
            return pd.DataFrame()

    # If Budget is specified, filter diamonds above that budget
    if "Budget" in constraints:
        user_budget = constraints["Budget"]
        df = df[df["Price"] <= user_budget]
        if df.empty:
            print(f"No diamonds found under price {user_budget}.")
            return pd.DataFrame()

    # Pre-filter by Carat if specified
    if "Carat" in constraints:
        # Set initial tolerance: 0.01 for labgrown, 0.05 for natural diamonds
        tolerance = 0.01 if constraints.get("Style", "").lower() == "labgrown" else 0.05
        df_carat = df[
            (df['Carat'] >= constraints["Carat"] - tolerance) &
            (df['Carat'] <= constraints["Carat"] + tolerance)
        ]
        
        # If no results found, relax the tolerance (e.g., double it) to show close matches
        if df_carat.empty:
            print(f"No exact carat matches found with tolerance {tolerance}. Relaxing tolerance...")
            relaxed_tolerance = tolerance * 2  # You can adjust this factor as needed
            df_carat = df[
                (df['Carat'] >= constraints["Carat"] - relaxed_tolerance) &
                (df['Carat'] <= constraints["Carat"] + relaxed_tolerance)
            ]
        
        # If df_carat now has results, build a temporary FAISS index on these diamonds
        if not df_carat.empty:
            subset_indices = df_carat.index.tolist()
            all_embeddings = np.load(EMBEDDING_FILE_PATH)
            subset_embeddings = all_embeddings[subset_indices]
            temp_index = faiss.IndexFlatL2(all_embeddings.shape[1])
            temp_index.add(subset_embeddings)
            new_top_k = min(top_k, len(df_carat))
            query_embedding = model.encode(user_query, convert_to_numpy=True)
            D, I = temp_index.search(np.array([query_embedding]), new_top_k)
            valid_indices = [i for i in I[0] if 0 <= i < len(df_carat)]
            valid_D = D[0][:len(valid_indices)]
            results_df = df_carat.iloc[valid_indices].copy()
            results_df['distance'] = valid_D
        else:
            # If still no matches, fall back to the full dataset
            query_embedding = model.encode(user_query, convert_to_numpy=True)
            new_top_k = min(top_k, df.shape[0])
            D, I = faiss_index.search(np.array([query_embedding]), new_top_k)
            valid_indices = [i for i in I[0] if 0 <= i < df.shape[0]]
            valid_D = D[0][:len(valid_indices)]
            results_df = df.iloc[valid_indices].copy()
            results_df['distance'] = valid_D
    else:
        query_embedding = model.encode(user_query, convert_to_numpy=True)
        new_top_k = min(top_k, df.shape[0])
        D, I = faiss_index.search(np.array([query_embedding]), new_top_k)
        valid_indices = [i for i in I[0] if 0 <= i < df.shape[0]]
        valid_D = D[0][:len(valid_indices)]
        results_df = df.iloc[valid_indices].copy()
        results_df['distance'] = valid_D

    # Composite ranking: compute a composite score for each candidate.
    def compute_score(row):
        score = row['distance']
        if "Carat" in constraints:
            score += 1000 * abs(row["Carat"] - constraints["Carat"])  # Increased weight to 1000
        # If the user gave a Budget, penalize diamonds that are far below that budget
        # so that higher-priced diamonds (closer to the budget) rank better.
        if "Budget" in constraints:
            user_budget = constraints["Budget"]
            # E.g. penalize the absolute difference from the budget
            # The smaller the difference, the lower the penalty => higher rank
            score += 0.05 * abs(row["Price"] - user_budget)
        else:
            # If no budget specified, revert to your old logic that penalizes higher price
            try:
                price = float(row["Price"])
            except:
                price = 0
            score += 0.1 * price

        # Penalties for mismatch in Clarity and Color
        for attr, penalty in [("Clarity", 50), ("Color", 50)]:
            if attr in constraints and row[attr] != constraints[attr]:
                score += penalty

        # Penalties for mismatch in Cut, Symmetry, Polish
        for attr, penalty in [("Cut", 20), ("Symmetry", 20), ("Polish", 20)]:
            if attr in constraints and row[attr] != constraints[attr]:
                score += penalty

        return score

    results_df['score'] = results_df.apply(compute_score, axis=1)
    results_df = results_df.sort_values(by='score', ascending=True)
    return results_df.head(5).reset_index(drop=True)

# ------------------- Groq Integration -------------------
def generate_groq_response(user_query, relevant_data, client):
    """
    Use FAISS retrieval to enhance the prompt before sending it to Groq.
    Sends a prompt to Groq's LLM to generate a shop assistant–style response using the provided diamond details.
    """
    prompt = f"""You are a friendly and knowledgeable shop assistant at a diamond store.
    If the user says hi or hello then respond accordingly introducing yourself.
Your goal is to help the customer find diamonds that best match their query.
Speak in a warm, engaging, and professional tone, and use only the provided diamond details.
Include all of the following details for each diamond option: Style, Carat, Price, Clarity, Color, Cut, Shape, Lab, Polish, and Symmetry.

User Query: {user_query}

Here are some diamond details that might be relevant:
{relevant_data}

Please provide a detailed, numbered recommendation (1 to 5) for the customer. 
In particular, if the diamond prices are significantly below the customer's specified budget, prioritize recommending the higher-priced options within the budget as these typically represent diamonds with enhanced quality and value. 
For each diamond option, include a full explanation of its attributes and why it might be a good choice. For example:
1. A 0.33 carat diamond with VS1 clarity, F color, excellent cut, and round shape, priced at $614.0. This diamond has an excellent polish and symmetry, and it's certified by the Gemological Institute of America (GIA).
2. Although slightly smaller, we also have a 0.32 carat diamond with VS1 clarity, F color, excellent cut, and round shape, priced at $488.0. This diamond also has an excellent polish and symmetry, and it's certified by the GIA.
Make sure your response is complete and includes a separate, detailed explanation for each diamond option on a separate line.

Please assist the customer by providing the diamond details (carat, clarity, color, cut, shape, price, and style) along with some helpful suggestions.
Tell them what diamond would be a better choice and why.
Now, generate a helpful and informative response while maintaining a professional, formal tone.
Please only use the above information to generate your response. If no relevant diamonds are found, politely inform the user.

Example Conversations:
Q: I need a 0.5-carat diamond with a G color.
A: We have multiple 0.5-carat diamonds in G color with different clarity grades. For example, one with VS1 clarity is priced at $2,500.

Q: What's the best cut for a solitaire ring?
A: The best cut for a solitaire ring is an Ideal or Excellent cut, as it maximizes brilliance.

Now, based on our diamond inventory, here are the most relevant options:
{relevant_data}


Answer:"""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=750
    )
    return chat_completion.choices[0].message.content

# ------------------- Main Chatbot Logic -------------------
def diamond_chatbot(user_query, df, faiss_index, model, client):
    # 1. Quick check for "hi" or "hello"
    if user_query.strip().lower() in ["hi", "hello"]:
        print("Hello! I'm your diamond assistant. How can I help you find the perfect diamond today?")
        return
    
    # 2. Extract constraints from the user query
    constraints = extract_constraints_from_query(user_query)

    # 3. If constraints are empty, handle that gracefully
    if not constraints:
        print("Hello! I'm your diamond assistant. Please let me know your preferred carat, clarity, color, " 
            "cut, or budget so I can help you find the perfect diamond.")
        return

    # 4. Otherwise, proceed with your existing logic
    results_df = hybrid_search(user_query, df, faiss_index, model, top_k=200)
    if results_df.empty:
        print("No matching diamonds found. Please try a different query.")
        return

    top_5 = results_df.head(5)
    relevant_data = "\n".join(top_5['combined_text'].tolist())
    groq_response = generate_groq_response(user_query, relevant_data, client)

    print(groq_response)
    
# ------------------- Main Execution -------------------
def main():
    
    # Load once outside the loop
    #df, index, model, client = setup_chatbot()
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Update with your key
    client = Groq()

    embedding_file = 'diamond_embeddings.npy'
    faiss_index_file = 'diamond_faiss_index.faiss'
    dataframe_file = 'diamond_dataframe.csv'
    model_path = 'sentence_transformer_model'
    file_path = 'diamonds.csv'

    try:
        df, embeddings, index, model = load_data_and_index(embedding_file, faiss_index_file, dataframe_file, model_path)
        print("Data, embeddings, and FAISS index loaded from disk.")
    except Exception as e:
        print("Error loading existing data:", e)
        print("Running first-time data load and creating index...")
        df, embeddings, index, model = data_and_embedding(file_path, embedding_file, faiss_index_file, dataframe_file, model_path)

    # Conversation loop: Continue until user types "exit" or "quit"
    while True:
        user_query = input("Hi! How can I help you? : ")
        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for visiting! Have a wonderful day.")
            break

        # If user said "hi" or "hello", let's just pass it directly to diamond_chatbot()
        # The fallback inside diamond_chatbot() will handle it.
        if user_query.strip().lower() in ["hi", "hello"]:
            diamond_chatbot(user_query, df, index, model, client)
            print("\n---\n")
            continue

        constraints = extract_constraints_from_query(user_query)
        if "Style" not in constraints:
            style_input = input("Please specify the style (LabGrown or Natural): ")
            user_query += " " + style_input

        diamond_chatbot(user_query, df, index, model, client)
        print("\n---\n")
    
if __name__ == "__main__":
    main()
import streamlit as st
import json
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# Load the JSON Data    
with open('Vital.json', 'r') as json_file:
    json_data = json.load(json_file)

# Vectorize the Data using TF-IDF
texts = [entry['desc'] for entry in json_data]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

for i, entry in enumerate(json_data):
    entry['vector'] = X[i].toarray().flatten().tolist()

# Check if Faiss index file exists, load it; otherwise, build the index
faiss_index_file = 'faiss_index.bin'
try:
    index = faiss.read_index(faiss_index_file)
except:
    # Set Up a Vector Search Engine
    index = faiss.IndexFlatL2(len(json_data[0]['vector']))
    vectors = np.array([entry['vector'] for entry in json_data])
    index.add(vectors)

    # Save the Faiss index
    faiss.write_index(index, faiss_index_file)

# Set Up OpenAI API Key
openai.api_key = OPEN_API_KEY

# Set the page configuration including the favicon
st.set_page_config(
    page_title="Chimera AI",
    page_icon="chimeraAI/chimera-logo.jpg",  # You can replace this with the path to your favicon
)

# Streamlit App
st.title("Chimera AI")

# User Input
user_query = st.text_input("query me!!:")

if user_query:
    # Perform Vector Search
    query_embedding = vectorizer.transform([user_query]).toarray().flatten().tolist()
    _, idx = index.search(np.array([query_embedding]), 1)
    relevant_entry = json_data[idx[0][0]]

    # Check if the Query Asks for Document URL
    if "document_url" in relevant_entry:
        document_url = relevant_entry["document_url"]
        response = f"The document URL for {relevant_entry['name']} is: {document_url}"
    else:
        # Use the "desc" field as the answer
        response = f"The answer for {relevant_entry['name']} is: {relevant_entry['desc']}."

    # Use GPT Turbo to Generate Response
    context = f"User Query: {user_query}\nJSON Entry: {relevant_entry}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
        ],
        temperature=0,
        max_tokens=100,
    )

    gpt_response = response['choices'][0]['message']['content']

    # Display Results
    st.subheader("Chimera AI:")
    st.write(gpt_response)

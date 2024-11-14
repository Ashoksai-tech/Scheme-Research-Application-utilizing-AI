import streamlit as st
import openai
import faiss
import pickle
import numpy as np
from langchain.document_loaders import UnstructuredURLLoader
from openai.embeddings_utils import get_embedding

# Set OpenAI API Key

openai.api_key = "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"

# Function to read URLs and fetch content
# Function to read URLs and fetch content
def load_content_from_url(url):
    try:
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        return documents[0].page_content if documents else "Content could not be loaded."
    except Exception as e:
        st.error(f"Error loading content from URL: {e}")
        return "Content could not be loaded."



# Summarize content based on specific categories
def summarize_content(content, category):
    prompt = f"Summarize the following content focusing on {category}: {content}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# Get embedding vector for text using OpenAI's embedding model
# Get embedding vector for text using OpenAI's embedding model
def get_embedding_vector(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


# Generate FAISS index and store it
def create_faiss_index(embedding_vectors):
    dimension = len(embedding_vectors[0])  # assuming all vectors have the same dimension
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embedding_vectors).astype("float32"))
    with open('faiss_store_openai.pkl', 'wb') as f:
        pickle.dump(index, f)
    return index

# Load FAISS index from file
def load_faiss_index():
    with open('faiss_store_openai.pkl', 'rb') as f:
        return pickle.load(f)

# Streamlit app
st.title("Scheme Research Tool")

# Sidebar for URL input
st.sidebar.header("Enter Scheme URL")
url_input = st.sidebar.text_input("URL:")
process_button = st.sidebar.button("Process URLs")

# If process button is clicked
if process_button and url_input:
    with st.spinner("Loading content..."):
        content = load_content_from_url(url_input)
        
    # Summarize scheme into key categories
    st.subheader("Scheme Summary")
    categories = ["Benefits", "Application Process", "Eligibility", "Documents Required"]
    summaries = {cat: summarize_content(content, cat) for cat in categories}
    
    # Display summarized information
    for cat, summary in summaries.items():
        st.write(f"**{cat}:** {summary}")
    
    # Generate embedding for each summary category and create FAISS index
    embeddings = [get_embedding_vector(summary) for summary in summaries.values()]
    index = create_faiss_index(embeddings)
    st.success("Content processed and indexed for query-based interaction.")

# Allow users to ask questions
st.subheader("Ask a Question about the Scheme")
query = st.text_input("Your Question:")
if query:
    query_vector = get_embedding_vector(query)
    index = load_faiss_index()
    
    distances, indices = index.search(np.array([query_vector]), k=1)
    response_category = list(summaries.keys())[indices[0][0]]
    
    st.write(f"**Answer from {response_category}:** {summaries[response_category]}")

# Save summary along with URL for future reference
if st.button("Save Summary"):
    with open("summary.txt", "w") as file:
        file.write(f"URL: {url_input}\n")
        for cat, summary in summaries.items():
            file.write(f"{cat}: {summary}\n")
    st.success("Summary saved successfully!")
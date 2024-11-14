import streamlit as st
from transformers import pipeline
from langchain.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Load facebook/bart-large-cnn for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load SBERT for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to split large content for summarization
def split_text(text, max_chunk_size=1024):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = ' '.join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

# Summarize content based on specific categories
def summarize_content(content, category):
    prompt = f"{category}: {content}"
    text_chunks = split_text(prompt)
    summaries = []
    for chunk in text_chunks:
        # Summarize each chunk using facebook/bart-large-cnn
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Load content from URL using LangChain
def load_content_from_url(url):
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()
    return documents[0].page_content if documents else ""

# Generate FAISS index from embedding vectors
def create_faiss_index(embedding_vectors):
    dimension = len(embedding_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embedding_vectors).astype("float32"))
    with open('faiss_store.pkl', 'wb') as f:
        pickle.dump(index, f)
    return index

# Load FAISS index from file
def load_faiss_index():
    with open('faiss_store.pkl', 'rb') as f:
        return pickle.load(f)

# Streamlit App Code
st.title("Scheme Research Tool")

# Sidebar for URL input
st.sidebar.header("Enter Scheme URL")
url_input = st.sidebar.text_input("URL:")
process_button = st.sidebar.button("Process URLs")

# Process URL and summarize content
if process_button and url_input:
    with st.spinner("Loading content..."):
        content = load_content_from_url(url_input)
        st.write("Content Loaded")

    # Summarize scheme into key categories
    st.subheader("Scheme Summary")
    categories = ["Benefits", "Application Process", "Eligibility", "Documents Required"]
    summaries = {cat: summarize_content(content, cat) for cat in categories}

    # Display summarized information
    for cat, summary in summaries.items():
        st.write(f"**{cat}:** {summary}")

    # Generate embeddings for summaries and create FAISS index
    embeddings = [embedder.encode(summary) for summary in summaries.values()]
    index = create_faiss_index(embeddings)
    st.success("Content processed and indexed for query-based interaction.")

# Allow users to ask questions
st.subheader("Ask a Question about the Scheme")
query = st.text_input("Your Question:")
if query:
    query_vector = embedder.encode(query)
    index = load_faiss_index()
    
    distances, indices = index.search(np.array([query_vector]), k=1)
    response_category = list(summaries.keys())[indices[0][0]]
    st.write(f"**Answer from {response_category}:** {summaries[response_category]}")

# Save summarized content
if st.button("Save Summary"):
    if summaries:
        with open("summary.txt", "w") as file:
            file.write(f"URL: {url_input}\n")
            for cat, summary in summaries.items():
                file.write(f"{cat}: {summary}\n")
        st.success("Summary saved successfully!")
    else:
        st.warning("Please process a URL first to generate summaries.")
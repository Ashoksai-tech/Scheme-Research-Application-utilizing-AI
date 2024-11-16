import streamlit as st
from langchain_community.document_loaders import FireCrawlLoader
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# FireCrawl API Key
FIRECRAWL_API_KEY = "fc-75104c867aa245519f9e0df2183d7781"

# Load DistilBART-cnn for summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load SBERT for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to fetch content using FireCrawlLoader
def fetch_content_with_firecrawl(url, mode="scrape", params=None):
    try:
        loader = FireCrawlLoader(
            api_key=FIRECRAWL_API_KEY,
            url=url,
            mode=mode,
            params=params
        )
        documents = loader.load()
        if not documents:
            return "No content extracted from the URL. Please check the URL or try another one."
        
        # Combine all loaded documents into a single content string
        content = "\n".join([doc.page_content for doc in documents])
        return content
    except Exception as e:
        return f"Error fetching content: {str(e)}"

# Extract specific sections from HTML content
def extract_sections(html_content):
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract Benefits
        benefits = soup.find(string=lambda text: "Benefits" in text or "benefit" in text.lower())
        benefits_content = benefits.find_next("p").get_text(strip=True) if benefits else "No information available for Benefits."

        # Extract Application Process
        app_process = soup.find("a", string=lambda text: "Account Opening" in text or "application" in text.lower())
        app_process_content = app_process.find_next("p").get_text(strip=True) if app_process else "No information available for Application Process."

        # Extract Eligibility
        eligibility = soup.find(string=lambda text: "Eligibility" in text or "eligible" in text.lower())
        eligibility_content = eligibility.find_next("p").get_text(strip=True) if eligibility else "No information available for Eligibility."

        # Extract Documents Required
        documents = soup.find("h2", string=lambda text: "e-Documents" in text or "documents required" in text.lower())
        documents_content = "\n".join([li.get_text(strip=True) for li in documents.find_next("ul").find_all("li")]) if documents else "No information available for Documents Required."

        return {
            "Benefits": benefits_content,
            "Application Process": app_process_content,
            "Eligibility": eligibility_content,
            "Documents Required": documents_content,
        }
    except Exception as e:
        return {
            "Benefits": "Error extracting Benefits: " + str(e),
            "Application Process": "Error extracting Application Process: " + str(e),
            "Eligibility": "Error extracting Eligibility: " + str(e),
            "Documents Required": "Error extracting Documents Required: " + str(e),
        }

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
process_button = st.sidebar.button("Process URL")

# Process the URL and extract sections
if process_button and url_input:
    with st.spinner("Fetching and processing content..."):
        # Use FireCrawl to fetch content
        content = fetch_content_with_firecrawl(url_input, mode="crawl")
        
        if "Error" in content or "No content" in content:
            st.error(content)
        else:
            st.write("Content Loaded")

            # Extract specific sections
            sections = extract_sections(content)

            # Embed external links within section descriptions
            sections["Scheme Details"] = f"This scheme is detailed comprehensively. [Learn more here]({url_input}). Additionally, view the [Continuation Document](https://pmjdy.gov.in/files/E-Documents/Continuation_of_PMJDY.pdf) and the [Mission Document](https://pmjdy.gov.in/files/E-Documents/PMJDY_BROCHURE_ENG.pdf)."

            # Summarize extracted content
            summaries = {key: summarizer(value, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                         if value and "No information" not in value else value
                         for key, value in sections.items()}

            # Display summarized information
            for key, summary in summaries.items():
                st.write(f"**{key}:** {summary}")

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
    response_category = list(sections.keys())[indices[0][0]]
    st.write(f"**Answer from {response_category}:** {sections[response_category]}")

# Save summarized content
if st.button("Save Summary"):
    with open("summary.txt", "w") as file:
        file.write(f"URL: {url_input}\n")
        for key, summary in summaries.items():
            file.write(f"{key}: {summary}\n")
    st.success("Summary saved successfully!")

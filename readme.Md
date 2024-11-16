### **Project Documentation: Scheme Research Tool**

 

### **1. Introduction**

#### **1.1 Project Overview**
The **Scheme Research Tool** is a web-based application that extracts, processes, and organizes information about government schemes. The tool presents categorized insights into **Benefits**, **Application Process**, **Eligibility**, and **Documents Required**, while also providing an interactive Q&A system.

#### **1.2 Objective**
- Simplify scheme-related information into accessible summaries.
- Enable users to interactively query the data for relevant details.
- Provide a flexible and scalable tool using pre-trained models.

#### **1.3 Limitations**
- **Non-Usage of OpenAI API**: 
  - **Reason**: The OpenAI API was not utilized as I had already used the free API quota.
  - **Implication**: While the chosen pre-trained models (DistilBART and SBERT) provided satisfactory results, OpenAI's **GPT-3.5** or **GPT-4** could have delivered even more refined and context-aware outputs due to their advanced capabilities in language understanding and summarization.
  - **Recommendation**: Incorporating OpenAI’s APIs in future iterations of this project would enhance the summarization accuracy and Q&A functionality.

---

### **2. System Architecture**

#### **2.1 Workflow Diagram**
1. **Input**: User provides the URL of the scheme.
2. **Data Extraction**: Content is extracted using the **FireCrawl API**.
3. **Content Parsing**: Extracted HTML is parsed with **BeautifulSoup**.
4. **Summarization**: Content is summarized using **DistilBART**.
5. **Embedding**: Vector embeddings are generated using **SBERT**.
6. **Search and Q&A**: Query handling and retrieval using **FAISS**.
7. **Output**: Categorized summaries and relevant responses to queries.

---

### **3. Features**

1. **FireCrawl API for Content Extraction**:
   - Crawls pages to extract information dynamically.
2. **Categorized Summarization**:
   - Organizes extracted content into **Benefits**, **Application Process**, **Eligibility**, and **Documents Required**.
3. **Interactive Q&A System**:
   - Enables users to query specific aspects of the scheme.
4. **External Links Integration**:
   - Incorporates important links directly into summaries for user convenience.
5. **Export Capability**:
   - Allows users to save summaries locally.

---

### **4. Implementation Details**

#### **4.1 Data Extraction**
- **Tool Used**: FireCrawl API
- **Process**:
  - FireCrawl retrieves web content in "crawl" mode to gather data from all accessible pages.
  - Extracted HTML is processed to identify relevant `<p>`, `<a>`, and `<ul>` tags.

#### **4.2 Content Parsing and Categorization**
- **Key Method**: Parsing focuses on finding text related to:
  - **Benefits**: From descriptive `<p>` tags or headings.
  - **Application Process**: Found under account-related or form links.
  - **Eligibility**: Extracted from descriptive lists or `<p>` tags.
  - **Documents Required**: Found in `<ul>` tags under "e-Documents" or similar sections.

#### **4.3 Summarization**
- **Model Used**: DistilBART (HuggingFace)
- **Process**:
  - Content chunks are summarized into concise outputs for each category.
  - External links (e.g., forms, brochures) are incorporated into summaries for completeness.

#### **4.4 Embedding and Q&A**
- **Embedding Tool**: SBERT (SentenceTransformer)
- **Workflow**:
  - Summaries are encoded into vector embeddings.
  - FAISS indexes the embeddings for efficient similarity searches.
  - Queries are matched with indexed embeddings to find the most relevant section.

---

### **5. Results and Insights**

1. **Summarized Outputs**:
   - Clear, categorized information presented for each section.
2. **Q&A Interaction**:
   - Provides accurate responses by mapping queries to the closest content.
3. **Embedding Search Efficiency**:
   - Sub-second query response time due to FAISS indexing.

---

### **6. Challenges and Limitations**

#### **6.1 Limitations of Pre-Trained Models**
- **Reason**: The project used DistilBART and SBERT instead of OpenAI's advanced GPT models.
- **Impact**:
  - Summarization and Q&A functionality were effective but lacked the nuanced understanding and contextual awareness of OpenAI’s APIs.
- **Solution**:
  - In future versions, OpenAI APIs (e.g., GPT-3.5 or GPT-4) should be used for enhanced performance.

#### **6.2 Excessive Content Extraction**
- **Issue**: FireCrawl extracted entire web pages, leading to irrelevant data.
- **Solution**: Focused parsing of `<p>`, `<ul>`, and `<a>` tags significantly reduced noise.

---

### **7. Future Enhancements**

1. **OpenAI API Integration**:
   - Leverage GPT-4 for superior summarization and Q&A performance.
2. **Multilingual Support**:
   - Add regional language processing for better accessibility.
3. **Dynamic Categorization**:
   - Use machine learning models to detect and categorize new types of information.
4. **Mobile App Development**:
   - Extend tool accessibility to mobile platforms.

---

### **8. Conclusion**

The Scheme Research Tool effectively streamlines the process of extracting, summarizing, and interacting with government scheme data. Despite the absence of OpenAI APIs, the project successfully delivers valuable insights using alternative tools like FireCrawl, DistilBART, and SBERT. Future enhancements, including OpenAI API integration, can elevate the tool's performance significantly.

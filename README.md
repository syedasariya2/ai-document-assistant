````markdown
# 📄 AI Document Assistant

A simple Streamlit RAG application that allows users to upload documents or load documents from Google Drive, extract their text, create chunks, generate embeddings, perform FAISS semantic search, and ask questions using Groq.

## Features

- PDF upload
- DOC/DOCX upload
- TXT upload
- Markdown (MD) upload
- Google Drive file/folder loading
- Separate extraction functions
- File name preservation
- Page number preservation where available
- Document information display
- Overlapping text chunks
- Chunk count display
- Sentence Transformer embeddings
- FAISS vector search
- Groq-powered answers
- Retrieved source display
- File name shown for every source
- Page number shown where available
- Retrieved text chunk shown below every answer
- Streamlit session state for processed documents
- Embeddings are created once per processed document set
- Sentence Transformer model is cached
- Groq API key stored in Streamlit Secrets

---

# Project Structure

```text
ai-document-assistant/
│
├── app.py
├── requirements.txt
└── README.md
````

---

# Technology Stack

* Python
* Streamlit
* PyMuPDF
* python-docx
* Sentence Transformers
* FAISS
* NumPy
* gdown
* Groq

---

# How the Application Works

The application follows this pipeline:

```text
Local Upload
     │
     │
     ├──────────────┐
     │              │
     ▼              ▼
PDF/DOC/TXT/MD   Google Drive
     │              │
     └──────┬───────┘
            ▼
      Text Extraction
            │
            ▼
        Text Chunks
            │
            ▼
   Sentence Transformer
            │
            ▼
        Embeddings
            │
            ▼
       FAISS Index
            │
            ▼
      User Question
            │
            ▼
 Question Embedding
            │
            ▼
     FAISS Search
            │
            ▼
   Relevant Chunks
            │
            ▼
        Groq LLM
            │
            ▼
          Answer
            │
            ▼
       Source Chunks
```

---

# Supported Documents

The application supports:

* `.pdf`
* `.doc`
* `.docx`
* `.txt`
* `.md`

## Important DOC Note

The application accepts `.doc` files, but Python's `python-docx` library is designed for modern `.docx` files.

For reliable legacy `.doc` extraction, the file may need to be converted to `.docx`.

---

# 1. Local Document Upload

Users can upload multiple documents through the Streamlit sidebar.

The application extracts the text and stores metadata:

```python
{
    "text": "...",
    "file_name": "policy.pdf",
    "page": 4
}
```

For formats without reliable page information:

```python
{
    "text": "...",
    "file_name": "notes.txt",
    "page": None
}
```

---

# 2. Google Drive

Users can paste a publicly accessible Google Drive file or folder link.

The application downloads supported files using `gdown`.

Supported Drive documents include:

* PDF
* DOC/DOCX
* TXT
* MD

The Drive documents then go through the same processing pipeline as local documents.

```text
Google Drive
     ↓
Download
     ↓
Extract
     ↓
Chunk
     ↓
Embed
     ↓
FAISS
```

The Google Drive documents are not processed using a separate RAG system.

---

# 3. Text Chunking

The extracted text is divided into smaller overlapping pieces.

Default settings:

```text
Chunk size = 800 characters
Overlap = 150 characters
```

For example:

```text
Chunk 1
AAAAAAAAAAAAAAAAAAAAAAAA

Chunk 2
              AAAAAAAAAAAAAAAAAAAAAAAA

Chunk 3
                            AAAAAAAAAAAAAAAAAAAAAAAA
```

The overlap helps preserve context between neighboring chunks.

Each chunk keeps its source metadata:

```python
{
    "text": "chunk text...",
    "file_name": "policy.pdf",
    "page": 5
}
```

---

# 4. Sentence Transformer Embeddings

The application uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

Every document chunk is converted into a numerical vector.

Example:

```text
Text Chunk
    ↓
Sentence Transformer
    ↓
Embedding Vector
```

The embeddings are stored in Streamlit session state.

---

# 5. FAISS

FAISS is used for semantic vector search.

The document embeddings are added to a FAISS index.

When a user asks a question:

```text
Question
   ↓
Sentence Transformer
   ↓
Question Embedding
   ↓
FAISS Search
   ↓
Top 3 Relevant Chunks
```

The application uses cosine similarity through normalized embeddings and an inner-product FAISS index.

---

# 6. Groq

After FAISS retrieves the most relevant chunks, those chunks are provided to the Groq LLM.

The model is instructed to:

* Answer using the retrieved context
* Avoid inventing information
* Say when the information is not found
* Use only the supplied document context

---

# 7. API Key

The Groq API key is NOT hardcoded in `app.py`.

Create a Streamlit Secrets file:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do not upload `secrets.toml` to GitHub.

Add this to `.gitignore`:

```text
.streamlit/secrets.toml
```

---

# 8. Avoiding Repeated Embeddings

The application uses Streamlit session state.

The following objects are kept during the session:

```python
st.session_state.documents
st.session_state.chunks
st.session_state.embeddings
st.session_state.faiss_index
st.session_state.processed_files
```

This means the application does not create new embeddings every time the user asks a question.

The embedding model itself is also cached:

```python
@st.cache_resource
def load_embedding_model():
    ...
```

Therefore, the Sentence Transformer model is loaded once and reused.

---

# 9. Source Display

After every answer, the application displays:

```text
Retrieved Sources

Source 1

File: employee_policy.pdf
Page: 7
Similarity: 0.8234

[Retrieved text chunk]


Source 2

File: employee_policy.pdf
Page: 12
Similarity: 0.7912

[Retrieved text chunk]
```

This makes it possible to see where the answer came from.

---

# Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Locally

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# Streamlit Cloud Deployment

## Step 1: Create GitHub Repository

Create a new GitHub repository.

Upload:

```text
app.py
requirements.txt
README.md
```

Do NOT upload:

```text
.streamlit/secrets.toml
```

---

## Step 2: Deploy on Streamlit Cloud

Open Streamlit Cloud and create a new application.

Select your GitHub repository.

Choose:

```text
Main file:
app.py
```

Then deploy the application.

---

## Step 3: Add Groq Secret

Open the application's Settings/Secrets section.

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Save the secret and restart/redeploy the application.

---

# Security

Never write the API key directly inside:

```python
app.py
```

Do NOT do this:

```python
client = Groq(
    api_key="gsk_xxxxxxxxx"
)
```

Instead, the application uses:

```python
api_key = st.secrets.get("GROQ_API_KEY")
```

---

# Current RAG Pipeline

The current application implements:

```text
Document Ingestion
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embeddings
        ↓
FAISS Vector Search
        ↓
Relevant Context
        ↓
Groq
        ↓
Answer
        ↓
Retrieved Sources
```

---

# Future Improvements

Possible improvements include:

* Persistent FAISS index
* Persistent document storage
* Better DOC support
* Hybrid keyword + vector search
* Reranking
* Chat history
* Multiple conversations
* Source highlighting
* Page-specific document previews
* Streaming Groq responses
* Better Google Drive authentication
* Google Docs export support
* Document deletion and management

```
```

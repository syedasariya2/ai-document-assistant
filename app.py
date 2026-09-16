```python
import os
import io
import hashlib
import tempfile

import streamlit as st
import fitz
import faiss
import gdown
import numpy as np

from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")

st.write(
    "Upload documents or load them from Google Drive, "
    "then ask questions about their content."
)


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_bytes, file_name):
    """
    Extract text from PDF page by page.
    Page numbers are preserved.
    """

    documents = []

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text()

        if text.strip():

            documents.append({
                "text": text,
                "file_name": file_name,
                "page": page_number
            })

    pdf.close()

    return documents


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_bytes, file_name):
    """
    Extract text from DOCX.

    DOCX files do not reliably expose page numbers
    through python-docx, so page is stored as None.
    """

    documents = []

    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    text = "\n".join(paragraphs)

    if text.strip():

        documents.append({
            "text": text,
            "file_name": file_name,
            "page": None
        })

    return documents


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file_bytes, file_name):
    """
    Extract text from TXT files.
    """

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    documents = []

    if text.strip():

        documents.append({
            "text": text,
            "file_name": file_name,
            "page": None
        })

    return documents


# ============================================================
# MARKDOWN EXTRACTION
# ============================================================

def extract_md(file_bytes, file_name):
    """
    Extract text from Markdown files.
    """

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    documents = []

    if text.strip():

        documents.append({
            "text": text,
            "file_name": file_name,
            "page": None
        })

    return documents


# ============================================================
# DOCUMENT EXTRACTION CONTROLLER
# ============================================================

def extract_document(file_bytes, file_name):
    """
    Select the correct extraction function.
    """

    extension = file_name.lower().split(".")[-1]

    if extension == "pdf":

        return extract_pdf(
            file_bytes,
            file_name
        )

    elif extension in ["doc", "docx"]:

        return extract_docx(
            file_bytes,
            file_name
        )

    elif extension == "txt":

        return extract_txt(
            file_bytes,
            file_name
        )

    elif extension == "md":

        return extract_md(
            file_bytes,
            file_name
        )

    return []


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    documents,
    chunk_size=800,
    overlap=150
):
    """
    Split document text into overlapping chunks.

    Each chunk keeps its source metadata.
    """

    chunks = []

    for document in documents:

        text = document["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "file_name": document["file_name"],
                    "page": document["page"]
                })

            start += chunk_size - overlap

    return chunks


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_embedding_model():
    """
    Load the embedding model only once.
    """

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks, model):
    """
    Convert all document chunks into embeddings.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    return embeddings


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):
    """
    Create a FAISS index using cosine similarity.
    """

    embeddings = embeddings.copy()

    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# DOCUMENT ID
# ============================================================

def create_document_id(file_name, file_bytes):
    """
    Create a unique ID for a document.
    This prevents the same document from being
    processed repeatedly.
    """

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return f"{file_name}_{file_hash}"


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(new_documents):

    if not new_documents:
        return

    with st.spinner(
        "Processing documents..."
    ):

        all_documents = (
            st.session_state.documents
            + new_documents
        )

        # Create chunks
        chunks = create_chunks(
            all_documents
        )

        st.session_state.chunks = chunks

        # Create embeddings
        model = load_embedding_model()

        embeddings = create_embeddings(
            chunks,
            model
        )

        st.session_state.embeddings = (
            embeddings
        )

        # Create FAISS index
        index = create_faiss_index(
            embeddings
        )

        st.session_state.faiss_index = (
            index
        )

        st.session_state.documents = (
            all_documents
        )

    st.success(
        f"Created {len(chunks)} document chunks."
    )


# ============================================================
# LOCAL FILE UPLOAD
# ============================================================

st.sidebar.header("📁 Local Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=[
        "pdf",
        "doc",
        "docx",
        "txt",
        "md"
    ],
    accept_multiple_files=True
)


if uploaded_files:

    new_documents = []

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        document_id = create_document_id(
            uploaded_file.name,
            file_bytes
        )

        if document_id not in st.session_state.processed_files:

            extracted = extract_document(
                file_bytes,
                uploaded_file.name
            )

            new_documents.extend(
                extracted
            )

            st.session_state.processed_files.add(
                document_id
            )

    if new_documents:

        process_documents(
            new_documents
        )


# ============================================================
# GOOGLE DRIVE
# ============================================================

st.sidebar.divider()

st.sidebar.header("☁️ Google Drive")

drive_link = st.sidebar.text_input(
    "Paste a public Google Drive file/folder link"
)

if st.sidebar.button(
    "Load from Google Drive"
):

    if not drive_link.strip():

        st.sidebar.warning(
            "Please paste a Google Drive link."
        )

    else:

        with st.spinner(
            "Loading files from Google Drive..."
        ):

            try:

                with tempfile.TemporaryDirectory() as temp_dir:

                    downloaded = gdown.download(
                        drive_link,
                        output=temp_dir,
                        quiet=True,
                        fuzzy=True
                    )

                    # ------------------------------------------------
                    # If a folder is downloaded
                    # ------------------------------------------------

                    files_to_process = []

                    if downloaded and os.path.isfile(downloaded):

                        files_to_process.append(
                            downloaded
                        )

                    else:

                        for root, dirs, files in os.walk(
                            temp_dir
                        ):

                            for file_name in files:

                                files_to_process.append(
                                    os.path.join(
                                        root,
                                        file_name
                                    )
                                )

                    new_documents = []

                    supported_extensions = [
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".txt",
                        ".md"
                    ]

                    for file_path in files_to_process:

                        file_name = os.path.basename(
                            file_path
                        )

                        extension = os.path.splitext(
                            file_name
                        )[1].lower()

                        if extension not in supported_extensions:
                            continue

                        with open(
                            file_path,
                            "rb"
                        ) as file:

                            file_bytes = file.read()

                        document_id = create_document_id(
                            file_name,
                            file_bytes
                        )

                        if document_id in st.session_state.processed_files:
                            continue

                        extracted = extract_document(
                            file_bytes,
                            file_name
                        )

                        new_documents.extend(
                            extracted
                        )

                        st.session_state.processed_files.add(
                            document_id
                        )

                    if new_documents:

                        process_documents(
                            new_documents
                        )

                        st.sidebar.success(
                            f"Loaded {len(new_documents)} document sections."
                        )

                    else:

                        st.sidebar.warning(
                            "No supported PDF, DOC/DOCX, TXT, or MD files were found."
                        )

            except Exception as e:

                st.sidebar.error(
                    f"Google Drive error: {e}"
                )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.documents:

    st.divider()

    st.subheader(
        "📄 Extracted Document Information"
    )

    unique_files = sorted(
        set(
            document["file_name"]
            for document in st.session_state.documents
        )
    )

    st.write(
        f"**Loaded files:** {len(unique_files)}"
    )

    st.write(
        f"**Created chunks:** "
        f"{len(st.session_state.chunks)}"
    )

    if st.session_state.embeddings is not None:

        st.write(
            f"**Embeddings:** "
            f"{len(st.session_state.embeddings)}"
        )

    for document in st.session_state.documents:

        page = document["page"]

        page_text = (
            str(page)
            if page is not None
            else "N/A"
        )

        with st.expander(
            f"{document['file_name']} | Page: {page_text}"
        ):

            st.write(
                document["text"][:3000]
            )


# ============================================================
# ASK QUESTIONS
# ============================================================

st.divider()

st.subheader(
    "💬 Ask Questions About Your Documents"
)

question = st.text_input(
    "Enter your question:"
)


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def search_documents(question, top_k=3):

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        question_embedding
    )

    distances, indices = (
        st.session_state.faiss_index.search(
            question_embedding,
            min(
                top_k,
                len(st.session_state.chunks)
            )
        )
    )

    results = []

    for score, index in zip(
        distances[0],
        indices[0]
    ):

        chunk = st.session_state.chunks[index]

        results.append({
            "text": chunk["text"],
            "file_name": chunk["file_name"],
            "page": chunk["page"],
            "score": float(score)
        })

    return results


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(question, results):

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        return (
            "GROQ_API_KEY was not found in "
            "Streamlit Secrets."
        )

    client = Groq(
        api_key=api_key
    )

    context_parts = []

    for result in results:

        page = (
            result["page"]
            if result["page"] is not None
            else "N/A"
        )

        context_parts.append(
            f"""
Source: {result['file_name']}
Page: {page}

{result['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY
the provided document context.

If the answer is not available in the
documents, clearly say that the information
was not found in the uploaded documents.

Do not invent information.

Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using "
                    "provided document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1
    )

    return response.choices[0].message.content


# ============================================================
# QUESTION ANSWERING
# ============================================================

if question:

    if not st.session_state.chunks:

        st.warning(
            "Please upload or load documents first."
        )

    else:

        with st.spinner(
            "Searching documents..."
        ):

            results = search_documents(
                question,
                top_k=3
            )

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                results
            )

        st.subheader(
            "🤖 Answer"
        )

        st.write(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        st.subheader(
            "📚 Retrieved Sources"
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            page = (
                result["page"]
                if result["page"] is not None
                else "N/A"
            )

            st.markdown(
                f"### Source {i}"
            )

            st.write(
                f"**File:** {result['file_name']}"
            )

            st.write(
                f"**Page:** {page}"
            )

            st.write(
                f"**Similarity:** "
                f"{result['score']:.4f}"
            )

            st.info(
                result["text"]
            )
```

---

## 2. `requirements.txt`

```text
streamlit
pymupdf
python-docx
sentence-transformers
faiss-cpu
numpy
groq
gdown
```

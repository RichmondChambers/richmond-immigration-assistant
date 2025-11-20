# index_builder.py

import os
import pickle
import numpy as np
import faiss
import openai
import docx
import PyPDF2

# 🔹 This is the folder in your Google Drive where your knowledge docs live.
# It can contain sub-folders; those will be indexed too.
KNOWLEDGE_DIR = "/content/drive/MyDrive/UK-Immigration-Knowledge"

# 🔹 These are the files your app already uses
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"


def extract_text_from_file(path: str) -> str:
    """
    Read a file from disk and return plain text.
    Supports: .txt, .md, .docx, .pdf
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    # TXT / MD
    if ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # DOCX
    if ext == ".docx":
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    # PDF
    if ext == ".pdf":
        text_chunks = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
        return "\n\n".join(text_chunks)

    # Unsupported type
    return ""


def split_into_chunks(text: str, max_chars: int = 1500, overlap: int = 200):
    """
    Very simple character-based chunking with overlap.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def embed_texts(texts, model="text-embedding-3-small", batch_size=64) -> np.ndarray:
    """
    Get embeddings for a list of texts using OpenAI embeddings API.
    Assumes openai.api_key has already been set in app.py.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(
            input=batch,
            model=model
        )
        for item in response.data:
            all_embeddings.append(item.embedding)

    return np.array(all_embeddings, dtype=np.float32)


def build_index_from_drive():
    """
    Scan KNOWLEDGE_DIR (including sub-folders), read all supported files,
    chunk, embed, and rebuild FAISS index + metadata.pkl.

    This is called by the Streamlit app when it starts.
    """
    all_chunks = []
    metadata = []

    if not os.path.isdir(KNOWLEDGE_DIR):
        print(f"[index_builder] Knowledge directory does not exist: {KNOWLEDGE_DIR}")
        # Create an empty index so the app doesn't crash
        dim = 1536  # text-embedding-3-small dimension
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump([], f)
        return

    # Walk through all files in the knowledge directory (including sub-folders)
    for root, dirs, files in os.walk(KNOWLEDGE_DIR):
        for filename in files:
            path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            # Only index supported types
            if ext not in [".txt", ".md", ".docx", ".pdf"]:
                continue

            text = extract_text_from_file(path)
            if not text.strip():
                continue

            chunks = split_into_chunks(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "content": chunk,
                    "file_path": path,
                    "file_name": filename,
                    "chunk_index": idx,
                })

    if not all_chunks:
        # No text found: build an empty index
        dim = 1536  # embedding size
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump([], f)
        print("[index_builder] No text found, created empty index.")
        return

    # Embed all chunks
    print(f"[index_builder] Embedding {len(all_chunks)} chunks...")
    embeddings = embed_texts(all_chunks, model="text-embedding-3-small")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index + metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[index_builder] Index rebuilt with {len(all_chunks)} chunks.")

#!/usr/bin/env python3

# backend/rag_setup.py
import json
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import text2jsonl

load_dotenv() # Ensure API key is loaded for embeddings

# --- Configuration ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
FAISS_INDEX_PATH = "faiss_index"
# Adjust chunk size and overlap based on your documents and model context window
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# --- End Configuration ---

def get_embeddings_model():
    """Initializes and returns the OpenAI embeddings model."""
    try:
        # Use a standard OpenAI embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Or text-embedding-ada-002
        # Test the embedding model (optional, requires API call)
        # embeddings.embed_query("test query")
        print("OpenAI embeddings model loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"Error initializing OpenAI embeddings model: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly in .env")
        return None

def jsonl_metadata_func(json_object: dict, metadata: dict) -> dict:
    """
    Extracts metadata from a JSON object (one line in the jsonl file).

    Args:
        json_object: The Python dictionary loaded from a single JSON line.
        metadata: Existing metadata (like 'source' and 'seq_num').

    Returns:
        Updated metadata dictionary.
    """
    # Example: Extract 'name', 'desc', and 'yaml' fields if they exist in your JSON lines
    metadata["name"] = json_object.get("name", None)
    metadata["desc"] = json_object.get("desc", None)
    metadata["yaml"] = json_object.get("yaml", None)
    return metadata

def build_vector_store(embeddings):
    """Loads docs, splits them, creates embeddings, and builds the FAISS store."""
    if not embeddings:
        print("Cannot build vector store without embeddings model.")
        return None

    print(f"Loading documents from: {KNOWLEDGE_BASE_DIR}")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
         print(f"Error: Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
         return None

    all_docs = []
    try:
        # --- Load .txt files ---
        print("Loading .txt files...")
        txt_loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            # Optional: Add encoding if your files aren't UTF-8
            # loader_kwargs={'encoding': 'utf-8'}
        )
        txt_docs = txt_loader.load()
        if txt_docs:
            print(f"Loaded {len(txt_docs)} .txt documents.")
            all_docs.extend(txt_docs)

        # --- Load .md files ---
        print("Loading .md files...")
        md_loader = DirectoryLoader(
             KNOWLEDGE_BASE_DIR,
             glob="**/*.md",
             loader_cls=TextLoader, # TextLoader often works okay for basic MD
             show_progress=True,
             use_multithreading=True
        )
        md_docs = md_loader.load()
        if md_docs:
            print(f"Loaded {len(md_docs)} .md documents.")
            all_docs.extend(md_docs)

        # --- Load .dat files ---
        print("Converting .dat files to *.jsonl...")
        for filename in os.listdir(KNOWLEDGE_BASE_DIR):
            if filename.endswith('.dat'):
                input_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                entries = text2jsonl.parse_custom_file(input_path)

                # Create output filename with .jsonl extension
                basename = os.path.splitext(filename)[0]
                output_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{basename}.jsonl")
                text2jsonl.write_jsonl(entries, output_path)

        # --- Load .jsonl files ---
        print("Loading .jsonl files...")
        # IMPORTANT: Configure jq_schema based on your JSON structure
        # '.' means the entire JSON object content becomes the document page_content
        # '.text_field' would mean only the value of the 'text_field' key is used
        # Adjust this schema to point to the main text content in your JSON lines.
        jsonl_jq_schema = '.'

        jsonl_loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR,
            glob="**/*.jsonl",
            loader_cls=JSONLoader,
            loader_kwargs={
                'jq_schema': jsonl_jq_schema,
                'json_lines': True, # Crucial for .jsonl
                'text_content': False, # Often set False when using jq_schema to avoid double processing
                # Optional: Add metadata extraction
                # 'metadata_func': jsonl_metadata_func
            },
            show_progress=True,
            # Multithreading might be less effective here depending on jq performance
            use_multithreading=True
        )
        jsonl_docs = jsonl_loader.load()
        if jsonl_docs:
            print(f"Loaded {len(jsonl_docs)} documents from .jsonl files.")
            all_docs.extend(jsonl_docs)
        # --- End JSONL Loading ---

        if not all_docs:
            print(f"No documents found in '{KNOWLEDGE_BASE_DIR}' with specified globs (.txt, .md, .jsonl).")
            return None

        print(f"Total documents loaded: {len(all_docs)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        splits = text_splitter.split_documents(all_docs)
        print(f"Split documents into {len(splits)} chunks.")

        if not splits:
             print("No chunks were created after splitting. Check document content and splitter settings.")
             return None

        print("Creating FAISS vector store...")
        # Ensure splits list is not empty before passing to FAISS
        if not splits:
             print("Error: No document chunks to add to FAISS.")
             return None
        vector_store = FAISS.from_documents(splits, embeddings)
        print("FAISS vector store created.")

        print(f"Saving FAISS index to: {FAISS_INDEX_PATH}")
        vector_store.save_local(FAISS_INDEX_PATH)
        print("FAISS index saved successfully.")
        return vector_store

    except ImportError:
         print("\nError: Missing dependencies for loading documents.")
         print("Please ensure 'jq' is installed (`pip install jq`) for JSONLoader.")
         print("Or check installs for other loaders if used (e.g., 'pypdf' for PDFs).")
         return None
    except Exception as e:
        print(f"Error building vector store: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for detailed error
        return None

def load_or_build_vector_store():
    """Loads the FAISS index if it exists, otherwise builds it."""
    embeddings = get_embeddings_model()
    if not embeddings:
        return None # Cannot proceed without embeddings

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from: {FAISS_INDEX_PATH}")
        try:
            # Allow dangerous deserialization is necessary for FAISS loading
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("FAISS index loaded successfully.")
            return vector_store
        except Exception as e:
            print(f"Error loading existing FAISS index: {e}. Rebuilding...")
            # Fall through to build if loading fails
            # Consider deleting the corrupted index folder here os.rmdir etc.
    else:
        print("No existing FAISS index found. Building new one...")

    # Build if not loaded
    vector_store = build_vector_store(embeddings)
    return vector_store

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Testing RAG setup...")
    vs = load_or_build_vector_store()
    if vs:
        print("\nVector store loaded/built. Testing retrieval...")
        try:
            retriever = vs.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 docs
            query = "How do you create a project?"
            results = retriever.invoke(query)
            print(f"\nResults for query '{query}':")
            if results:
                for i, doc in enumerate(results):
                    print(f"--- Result {i+1} ---")
                    print(f"Content: {doc.page_content[:200]}...") # Show snippet
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
            else:
                print("No relevant documents found.")
        except Exception as e:
            print(f"Error during retrieval test: {e}")
    else:
        print("\nFailed to load or build vector store.")

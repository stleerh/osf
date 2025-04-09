#!/usr/bin/env python3

# backend/rag_setup.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

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

def build_vector_store(embeddings):
    """Loads docs, splits them, creates embeddings, and builds the FAISS store."""
    if not embeddings:
        print("Cannot build vector store without embeddings model.")
        return None

    print(f"Loading documents from: {KNOWLEDGE_BASE_DIR}")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
         print(f"Error: Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
         return None

    try:
        # Using TextLoader for .txt and DirectoryLoader with glob for flexibility
        # You might need different loaders (e.g., PyPDFLoader) for other file types
        loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR,
            glob="**/*.txt", # Load .txt files
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True
        )
        # Add more loaders if needed, e.g., for markdown
        md_loader = DirectoryLoader(
             KNOWLEDGE_BASE_DIR,
             glob="**/*.md", # Load .md files
             loader_cls=TextLoader, # TextLoader often works okay for basic MD
             show_progress=True,
             use_multithreading=True
        )

        docs = loader.load()
        md_docs = md_loader.load()
        all_docs = docs + md_docs # Combine documents from different loaders

        if not all_docs:
            print("No documents found in the knowledge base directory.")
            return None

        print(f"Loaded {len(all_docs)} documents.")

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
        vector_store = FAISS.from_documents(splits, embeddings)
        print("FAISS vector store created.")

        print(f"Saving FAISS index to: {FAISS_INDEX_PATH}")
        vector_store.save_local(FAISS_INDEX_PATH)
        print("FAISS index saved successfully.")
        return vector_store

    except Exception as e:
        print(f"Error building vector store: {e}")
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
            query = "What collects logs?"
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

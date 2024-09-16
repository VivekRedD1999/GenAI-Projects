# scripts/search.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def load_faiss_index(index_file='models/faiss.index'):
    """
    Loads the FAISS index from the specified file.

    Args:
        index_file (str): Path to the FAISS index file.

    Returns:
        faiss.Index: Loaded FAISS index object.
    """
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file not found at {index_file}.")
    index = faiss.read_index(index_file)
    print(f"Loaded FAISS index from {index_file}")
    return index

def load_segments(segments_file='models/segments.txt'):
    """
    Loads the segment names from the specified file.

    Args:
        segments_file (str): Path to the segments .txt file.

    Returns:
        list: List of segment names.
    """
    if not os.path.exists(segments_file):
        raise FileNotFoundError(f"Segments file not found at {segments_file}.")
    with open(segments_file, 'r', encoding='utf-8') as f:
        segments = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(segments)} segments.")
    return segments

def load_text_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads the Sentence Transformer model for generating query embeddings.

    Args:
        model_name (str): Pretrained Sentence Transformer model name.

    Returns:
        SentenceTransformer: Loaded Sentence Transformer model.
    """
    try:
        text_model = SentenceTransformer(model_name)
        print(f"Loaded text model: {model_name}")
        return text_model
    except Exception as e:
        raise RuntimeError(f"Error loading text model: {e}")

def search_query(query, text_model, faiss_index, segments, top_k=2):
    """
    Searches for the top_k most relevant segments to the query.

    Args:
        query (str): User input search query.
        text_model (SentenceTransformer): Loaded Sentence Transformer model.
        faiss_index (faiss.Index): Loaded FAISS index.
        segments (list): List of segment names.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: List of top_k segment names.
    """
    # Generate query embedding
    query_embedding = text_model.encode([query]).astype('float32')
    print(f"Query Embedding Shape: {query_embedding.shape}")  # Should be (1, 384)

    # Normalize embedding
    faiss.normalize_L2(query_embedding)

    # Search in FAISS index
    D, I = faiss_index.search(query_embedding, top_k)
    print(f"Distances: {D}")
    print(f"Indices: {I}")

    # Retrieve segments
    results = [segments[idx] for idx in I[0]]
    return results

if __name__ == "__main__":
    # Example usage
    query = "Discuss artificial intelligence applications in healthcare"
    
    # Load FAISS index and segments
    faiss_index = load_faiss_index()
    segments = load_segments()
    
    # Load text model
    text_model = load_text_model()
    
    # Perform search
    results = search_query(query, text_model, faiss_index, segments, top_k=2)
    print("Top relevant segments:")
    for res in results:
        print(res)

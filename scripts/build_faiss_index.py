# scripts/build_faiss_index.py

import faiss
import numpy as np

def build_faiss_index(text_embeddings_file='models/text_embeddings.npy', 
                      index_file='models/faiss.index', 
                      segments_file='models/segments.txt'):
    """
    Builds a FAISS index using only text embeddings.
    
    Args:
        text_embeddings_file (str): Path to text embeddings .npy file.
        index_file (str): Path to save the FAISS index.
        segments_file (str): Path to the segments .txt file.
    
    Returns:
        faiss.Index: Loaded FAISS index object.
        list: List of segment names.
    """
    # Load text embeddings
    embeddings = np.load(text_embeddings_file).astype('float32')
    print(f"Loaded {embeddings.shape[0]} text embeddings with dimension {embeddings.shape[1]}.")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Determine embedding dimension
    dimension = embeddings.shape[1]

    # Initialize FAISS index for Inner Product (cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    print(f"Initialized FAISS IndexFlatIP with dimension {dimension}.")

    # Add embeddings to index
    index.add(embeddings)
    print(f"Added {embeddings.shape[0]} embeddings to the FAISS index.")

    # Save the index
    faiss.write_index(index, index_file)
    print(f"FAISS index built and saved to {index_file}")

    # Load segments list
    with open(segments_file, 'r', encoding='utf-8') as f:
        segments = [line.strip() for line in f.readlines()]

    return index, segments

if __name__ == "__main__":
    build_faiss_index()

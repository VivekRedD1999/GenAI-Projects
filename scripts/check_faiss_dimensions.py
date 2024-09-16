# scripts/check_faiss_dimensions.py

import faiss

def check_faiss_dimensions(index_file='models/faiss.index'):
    index = faiss.read_index(index_file)
    print(f"FAISS Index Dimension: {index.d}")  # Expected: 384

if __name__ == "__main__":
    check_faiss_dimensions()

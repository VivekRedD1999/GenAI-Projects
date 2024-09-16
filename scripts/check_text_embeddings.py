# scripts/check_text_embeddings.py

import numpy as np

def check_text_embeddings(embeddings_file='models/text_embeddings.npy'):
    embeddings = np.load(embeddings_file)
    print(f"Text Embeddings Shape: {embeddings.shape}")  # Should be (num_segments, 384) for 'all-MiniLM-L6-v2'

if __name__ == "__main__":
    check_text_embeddings()

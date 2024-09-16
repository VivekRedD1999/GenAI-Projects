# scripts/check_combined_embeddings.py

import numpy as np

def check_combined_embeddings(embeddings_file='models/combined_embeddings.npy'):
    embeddings = np.load(embeddings_file)
    print(f"Combined Embeddings Shape: {embeddings.shape}")  # Should be (num_segments, 1152) if combined

if __name__ == "__main__":
    check_combined_embeddings()

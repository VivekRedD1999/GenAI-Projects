# combine_embeddings.py

import numpy as np
import os

def load_embeddings(embeddings_file, segments_file):
    embeddings = np.load(embeddings_file)
    with open(segments_file, 'r') as f:
        segments = [line.strip() for line in f.readlines()]
    return embeddings, segments

def combine_embeddings(text_embeddings_file='models/text_embeddings.npy', text_segments_file='models/segments.txt',
                      image_embeddings_file='models/image_embeddings.npy', image_segments_file='models/image_segments.txt',
                      combined_embeddings_file='models/combined_embeddings.npy',
                      combined_segments_file='models/combined_segments.txt'):
    # Load text embeddings
    text_embeddings, text_segments = load_embeddings(text_embeddings_file, text_segments_file)
    
    # Load image embeddings
    image_embeddings, image_segments = load_embeddings(image_embeddings_file, image_segments_file)
    
    # Ensure that segments match
    if set(text_segments) != set(image_segments):
        raise ValueError("Mismatch between text and image segments")
    
    # Sort segments to ensure alignment
    sorted_segments = sorted(text_segments)
    text_indices = [text_segments.index(seg) for seg in sorted_segments]
    image_indices = [image_segments.index(seg) for seg in sorted_segments]
    
    sorted_text_embeddings = text_embeddings[text_indices]
    sorted_image_embeddings = image_embeddings[image_indices]
    
    # Combine embeddings (e.g., concatenate)
    combined_embeddings = np.concatenate((sorted_text_embeddings, sorted_image_embeddings), axis=1)
    
    # Save combined embeddings and segments
    np.save(combined_embeddings_file, combined_embeddings)
    with open(combined_segments_file, 'w') as f:
        for seg in sorted_segments:
            f.write(seg + '\n')
    
    print(f"Combined embeddings saved to {combined_embeddings_file} and segments to {combined_segments_file}")

if __name__ == "__main__":
    combine_embeddings()


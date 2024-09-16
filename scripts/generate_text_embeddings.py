# scripts/generate_text_embeddings.py

from sentence_transformers import SentenceTransformer
import os
import numpy as np
from tqdm import tqdm

def generate_text_embeddings(transcript_dir='data/transcripts', 
                             model_name='all-MiniLM-L6-v2', 
                             embeddings_file='models/text_embeddings.npy',
                             segments_file='models/segments.txt'):
    # Load model
    text_model = SentenceTransformer(model_name)
    print(f"Loaded text model: {model_name}")

    # Read transcripts
    segments = []
    transcripts = []
    transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]

    for file in transcript_files:
        segment = os.path.splitext(file)[0]
        with open(os.path.join(transcript_dir, file), 'r', encoding='utf-8') as f:
            transcript = f.read()
            segments.append(segment)
            transcripts.append(transcript)

    # Generate embeddings
    print("Generating text embeddings...")
    embeddings = text_model.encode(transcripts, show_progress_bar=True)

    # Save embeddings and segment mapping
    np.save(embeddings_file, embeddings)
    with open(segments_file, 'w', encoding='utf-8') as f:
        for seg in segments:
            f.write(seg + '\n')

    print(f"Generated and saved text embeddings to {embeddings_file}")
    print(f"Saved segment mapping to {segments_file}")

if __name__ == "__main__":
    generate_text_embeddings()

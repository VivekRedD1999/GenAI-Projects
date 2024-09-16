# app.py

from flask import Flask, render_template, request, send_from_directory, abort
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

app = Flask(__name__)

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = os.path.join('models', 'faiss.index')
SEGMENTS_FILE = os.path.join('models', 'segments.txt')
SEGMENTS_DIR = os.path.join('static', 'segments')
THUMBNAILS_DIR = os.path.join('static', 'thumbnails')

# Load FAISS index
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index file not found at {FAISS_INDEX_PATH}. Please run the indexing script first.")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Load segments
if not os.path.exists(SEGMENTS_FILE):
    raise FileNotFoundError(f"Segments file not found at {SEGMENTS_FILE}.")
with open(SEGMENTS_FILE, 'r', encoding='utf-8') as f:
    segments = [line.strip() for line in f.readlines()]

# Load Sentence Transformer model
model = SentenceTransformer(MODEL_NAME)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query == '':
            return render_template('index.html', error="Please enter a valid query.", query=query)
        
        # Generate query embedding
        query_embedding = model.encode([query]).astype('float32')
        
        # Normalize embedding
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        top_k = 2
        D, I = faiss_index.search(query_embedding, top_k)
        
        # Retrieve top_k segments
        results = [segments[idx] for idx in I[0]]
        
        return render_template('results.html', query=query, results=results)
    
    return render_template('index.html')

@app.route('/video/<segment_name>')
def video(segment_name):
    video_filename = f"{segment_name}.mp4"
    video_path = os.path.join(SEGMENTS_DIR, video_filename)
    if os.path.exists(video_path):
        return send_from_directory(SEGMENTS_DIR, video_filename)
    else:
        abort(404)

@app.route('/thumbnail/<segment_name>')
def thumbnail(segment_name):
    thumbnail_filename = f"{segment_name}.jpg"
    thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_filename)
    if os.path.exists(thumbnail_path):
        return send_from_directory(THUMBNAILS_DIR, thumbnail_filename)
    else:
        abort(404)

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

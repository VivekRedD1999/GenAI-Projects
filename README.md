# Video Segment Search with FAISS and Transformers

## Description
A Flask-based application that utilizes FAISS for efficient searching of relevant video segments based on user queries. The system processes and indexes video segments, allowing users to retrieve and view the most pertinent segments seamlessly.

## Features
- **Search Functionality:** Retrieve the most relevant video segments across all indexed videos.
- **Video Playback:** Integrated video player for viewing selected segments directly on the web interface.
- **Efficient Indexing:** Utilizes FAISS for fast and scalable similarity searches.

## Project Structure

video-search-system/ 
├── app.py 
├── models/ 
│ ├── faiss.index 
│ ├── segments.txt 
│ └── text_embeddings.npy 
├── static/ 
│ ├── css/ 
│ │ └── styles.css 
│ ├── thumbnails/ 
│ │ ├── segment_0_30.jpg 
│ │ ├── segment_30_60.jpg 
│ │ └── ... 
│ └── segments/ 
│ ├── segment_0_30.mp4 
│ ├── segment_30_60.mp4 
│ └── ... 
├── templates/ 
│ ├── index.html 
│ ├── results.html 
│ └── 404.html 
├── scripts/ 
│ ├── build_faiss_index.py 
│ ├── generate_embeddings.py 
│ └── ... 
├── requirements.txt 
├── README.md 
└── .gitignore

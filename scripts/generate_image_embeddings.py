# generate_image_embeddings.py

import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

def generate_image_embeddings(frames_dir='data/frames', model_name='vit_b_16', embeddings_file='models/image_embeddings.npy'):
    # Load pre-trained ViT model
    if model_name == 'vit_b_16':
        image_model = models.vit_b_16(pretrained=True)
    else:
        raise ValueError("Model not supported")
    image_model.eval()
    
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    
    # List frames
    frame_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    segments = []
    image_embeddings = []
    
    for frame_path in frame_files:
        img = Image.open(frame_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            embedding = image_model(input_tensor)
            embedding = embedding.squeeze(0).numpy()  # Remove batch dimension
            image_embeddings.append(embedding)
            segment_name = os.path.splitext(os.path.basename(frame_path))[0].rsplit('_frame_', 1)[0]
            segments.append(segment_name)
    
    # Convert to NumPy array
    image_embeddings = np.array(image_embeddings)
    
    # Aggregate embeddings per segment (e.g., average if multiple frames per segment)
    unique_segments = list(set(segments))
    aggregated_embeddings = []
    for seg in unique_segments:
        indices = [i for i, s in enumerate(segments) if s == seg]
        seg_embeddings = image_embeddings[indices]
        avg_embedding = np.mean(seg_embeddings, axis=0)
        aggregated_embeddings.append(avg_embedding)
    
    # Save embeddings and segment mapping
    np.save(embeddings_file, aggregated_embeddings)
    with open('models/image_segments.txt', 'w') as f:
        for seg in unique_segments:
            f.write(seg + '\n')
    
    print(f"Generated and saved image embeddings to {embeddings_file} and segment mapping to models/image_segments.txt")

if __name__ == "__main__":
    generate_image_embeddings()

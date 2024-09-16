# extract_frames.py

from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import os
import numpy as np

def extract_key_frames(segment_path, frames_per_segment=6, output_dir='data/frames'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = VideoFileClip(segment_path)
    duration = video.duration
    timestamps = np.linspace(0, duration, num=frames_per_segment + 2)[1:-1]  # Exclude first and last
    frames = []
    for idx, t in enumerate(timestamps):
        frame = video.get_frame(t)
        image = Image.fromarray(frame)
        frame_filename = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(segment_path))[0]}_frame_{idx}.jpg"
        )
        image.save(frame_filename)
        frames.append(frame_filename)
    video.close()
    return frames

if __name__ == "__main__":
    segment_dir = 'data/segments'
    frame_dir = 'data/frames'
    segment_files = [os.path.join(segment_dir, f) for f in os.listdir(segment_dir) if f.endswith('.mp4')]
    for sf in segment_files:
        frames = extract_key_frames(sf)
        print(f"Extracted frames for {sf}: {frames}")

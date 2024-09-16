# segment_videos.py

from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def segment_video(video_path, segment_length=30, output_dir='data/segments'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = VideoFileClip(video_path)
    duration = int(video.duration)
    segments = []
    
    for start in range(0, duration, segment_length):
        end = min(start + segment_length, duration)
        segment = video.subclip(start, end)
        segment_filename = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(video_path))[0]}_segment_{start}_{end}.mp4"
        )
        segment.write_videofile(segment_filename, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        segments.append(segment_filename)
    
    video.close()
    return segments

if __name__ == "__main__":
    video_paths = ['data/video1.mp4', 'data/video2.mp4']
    for vp in video_paths:
        segment_videos = segment_video(vp)
        print(f"Segments created for {vp}: {segment_videos}")

# transcribe_audio_english.py

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import os

def transcribe_audio(segment_path, model, processor, device='cpu'):
    # Load audio
    speech, sr = librosa.load(segment_path, sr=16000)
    
    # Process audio
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription.lower()

def transcribe_segments(segments, model, processor, device='cpu', transcript_dir='data/transcripts'):
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)
    
    transcripts = {}
    for segment in segments:
        transcription = transcribe_audio(segment, model, processor, device)
        segment_name = os.path.splitext(os.path.basename(segment))[0]
        transcript_path = os.path.join(transcript_dir, f"{segment_name}.txt")
        with open(transcript_path, 'w') as f:
            f.write(transcription)
        transcripts[segment] = transcription
        print(f"Transcribed {segment}: {transcription}")
    return transcripts

if __name__ == "__main__":
    # Load English Wav2Vec 2.0 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    model.eval()
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Get list of segments
    segment_dir = 'data/segments'
    segments = [os.path.join(segment_dir, f) for f in os.listdir(segment_dir) if f.endswith('.mp4')]
    
    # Transcribe segments
    transcripts = transcribe_segments(segments, model, processor, device)

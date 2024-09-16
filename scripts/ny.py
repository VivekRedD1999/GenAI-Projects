# scripts/update_segments.py

import os

def update_segments_file(original_file='models/segments.txt', updated_file='models/segments_updated.txt'):
    with open(original_file, 'r', encoding='utf-8') as f_in, open(updated_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if '_' in line:
                video_id, segment_name = line.split('_', 1)
                f_out.write(f"{video_id},{segment_name}\n")
            else:
                raise ValueError(f"Invalid segment format: {line}. Expected 'video_id_segmentname'.")
    print(f"Updated segments saved to {updated_file}")

if __name__ == "__main__":
    update_segments_file()

from frame_extractor import extract_frames

video_path = "data/raw/fake_videos/sample.mp4"
output_dir = "data/frames/fake"

extract_frames(video_path, output_dir, frame_skip=20)

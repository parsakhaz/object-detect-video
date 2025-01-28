# Moondream Video Ad Detector

This tool automatically detects and highlights advertisements in videos using the Moondream2 vision-language model. It processes videos frame by frame and draws bounding boxes around detected advertisements.

## Features

- Real-time advertisement detection in videos
- Temporal smoothing for stable bounding boxes
- Frame-by-frame processing with IoU tracking
- Batch processing of multiple videos
- Web-compatible output format
- Configurable detection parameters

## Requirements

- Python 3.8+
- OpenCV (cv2)
- PyTorch
- Transformers
- Pillow (PIL)
- tqdm
- ffmpeg

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch transformers opencv-python pillow tqdm einops pyvips accelerate
```
3. Install ffmpeg:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

1. Create an `inputs` directory in the same folder as the script:
```bash
mkdir inputs
```

2. Place your video files in the `inputs` directory. Supported formats:
   - .mp4
   - .avi
   - .mov
   - .mkv
   - .webm

3. Run the script:
```bash
python main.py
```

### Optional Arguments:
- `--test`: Process only first 3 seconds of each video
```bash
python main.py --test
```

- `--preset`: Choose FFmpeg encoding preset (affects output quality vs. speed)
```bash
python main.py --preset ultrafast  # Fastest, lower quality
python main.py --preset veryslow   # Slowest, highest quality
```

## Detection Parameters

The script uses several parameters for optimal ad detection:

- Frame Processing: Every frame is analyzed for advertisements
- IoU Threshold: 0.5 (standard threshold for box matching)
- EMA Smoothing: Alpha = 0.3 (balance between smoothness and responsiveness)
- Temporal Window: 5 frames (for smoothing box movements)
- Full-screen Filter: Ignores detections covering >98% of both dimensions

## Output

Processed videos will be saved in the `outputs` directory with the format:
`detected_[original_filename]_web.mp4`

The output videos will include:
- Original video content
- Red bounding boxes around detected advertisements
- Box labels showing "Advertisement"
- Temporally smoothed detections
- Web-compatible H.264 encoding

## Notes

- Processing time depends on video length and GPU availability
- GPU is strongly recommended for faster processing
- Requires sufficient disk space for temporary files
- Detection quality may vary based on advertisement type and video quality
- Temporal smoothing helps reduce jitter in bounding boxes 
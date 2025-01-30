# Moondream Video Object Censor

This tool automatically detects and censors objects in videos using the Moondream2 vision-language model. It can censor any object type that Moondream2 can recognize by placing black boxes over detected regions.

## Features

- Real-time object detection and censoring in videos
- Grid-based detection for improved accuracy on large frames
- Flexible object type censoring
- Temporal smoothing for stable censor boxes
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
- numpy

## Installation

1. Clone this repository and create a new virtual environment
~~~bash
git clone https://github.com/parsakhaz/object-detect-video.git
python -m venv .venv
source .venv/bin/activate
~~~
2. Install the required packages:
~~~bash
pip install -r requirements.txt
~~~
3. Install ffmpeg:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg libvips`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

1. Create an `inputs` directory in the same folder as the script:
~~~bash
mkdir inputs
~~~

2. Place your video files in the `inputs` directory. Supported formats:
   - .mp4
   - .avi
   - .mov
   - .mkv
   - .webm

3. Run the script:
~~~bash
python main.py
~~~

### Optional Arguments:
- `--test`: Process only first 3 seconds of each video
~~~bash
python main.py --test
~~~

- `--preset`: Choose FFmpeg encoding preset (affects output quality vs. speed)
~~~bash
python main.py --preset ultrafast  # Fastest, lower quality
python main.py --preset veryslow   # Slowest, highest quality
~~~

- `--detect`: Specify what object type to censor
~~~bash
python main.py --detect person     # Censor people
python main.py --detect car        # Censor cars
python main.py --detect face       # Censor faces (default)
~~~

- `--rows` and `--cols`: Enable grid-based detection by splitting frames
~~~bash
python main.py --rows 2 --cols 2   # Split each frame into 2x2 grid
python main.py --rows 3 --cols 3   # Split each frame into 3x3 grid
python main.py --rows 2 --cols 4 --detect face   # Split each frame into 2x4 grid and censor faces
~~~

You can combine arguments:
~~~bash
python main.py --detect person --test --preset fast --rows 2 --cols 2
~~~

### Grid-based Detection

The grid-based detection feature splits each frame into smaller tiles before processing. This can improve detection accuracy by:
- Allowing the model to focus on smaller regions
- Better detecting small objects that might be missed in full-frame analysis
- Reducing the impact of scale on detection performance

The trade-offs are:
- Increased processing time (proportional to number of grid cells)
- Potential for duplicate detections (handled by NMS merging)
- Memory usage increases with grid size

## Output

Processed videos will be saved in the `outputs` directory with the format:
`censor_[object_type]_[original_filename].mp4`

The output videos will include:
- Original video content
- Black censor boxes over detected objects
- Web-compatible H.264 encoding

## Notes

- Processing time depends on video length, grid size, and GPU availability
- GPU is strongly recommended for faster processing
- Requires sufficient disk space for temporary files
- Detection quality may vary based on object type and video quality
- Temporal smoothing helps reduce jitter in censor boxes while maintaining responsiveness
- Detection accuracy depends on Moondream2's ability to recognize the specified object type
- Grid size should be chosen based on video resolution and object size 
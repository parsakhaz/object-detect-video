# Promptable Video Redaction with Moondream

This tool uses Moondream 2B, a powerful yet lightweight vision-language model, to detect and redact objects from videos. Moondream can recognize a wide variety of objects, people, text, and more with high accuracy while being much smaller than traditional models. The tool also provides comprehensive visualization capabilities for analyzing detection patterns.

[Try it now.](https://huggingface.co/spaces/moondream/promptable-video-redaction)

## About Moondream

Moondream is a tiny yet powerful vision-language model that can analyze images and answer questions about them. It's designed to be lightweight and efficient while maintaining high accuracy. Some key features:

- Only 2B parameters
- Fast inference with minimal resource requirements
- Supports CPU and GPU execution
- Open source and free to use
- Can detect almost anything you can describe in natural language

Links:

- [GitHub Repository](https://github.com/vikhyat/moondream)
- [Hugging Face](https://huggingface.co/vikhyatk/moondream2)
- [Build with Moondream](http://docs.moondream.ai/)

## Features

- Real-time object detection in videos using Moondream
- Multiple visualization styles:
  - Censor: Black boxes over detected objects
  - Bounding Box: Traditional bounding boxes with labels
  - Hitmarker: Call of Duty style crosshair markers
  - SAM: Segment Anything Model segmentation
  - Fast SAM: Faster but less detailed segmentation
  - Fuzzy-blur: Gaussian blur effect over detected objects
  - Pixelated-blur: Pixelation with blur effect
  - Intense-pixelated-blur: Stronger pixelation and blur
  - Obfuscated-pixel: Advanced pixelation with background blending
- Intelligent scene detection and tracking:
  - Automatic scene change detection
  - DeepSORT tracking with scene-aware reset
  - Persistent object tracking across frames
  - Smart tracker reset at scene boundaries
- Optional grid-based detection for improved accuracy
- Flexible object type detection using natural language
- Frame-by-frame processing with IoU-based merging
- Batch processing of multiple videos
- Web-compatible output format
- User-friendly web interface with real-time visualization
- Command-line interface for automation
- Detection data persistence and comprehensive visualization tools
- Support for test mode (process only first 3 seconds)
- Configurable FFmpeg encoding presets
- Advanced detection analysis with multiple visualization plots

## Requirements

### Python Dependencies

For Windows users, before installing other requirements, first install PyTorch with CUDA support:
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- Transformers 4.36+
- OpenCV 4.8+
- Gradio 4.0+
- FFmpeg-python
- Pillow 10.0+
- NumPy 1.24+
- Pandas 2.0+
- Matplotlib 3.7+
- Plotly
- Segment Anything Model (SAM) dependencies

### System Requirements

- FFmpeg (required for video processing)
- libvips (required for image processing)

Installation by platform:

- Ubuntu/Debian: `sudo apt-get install ffmpeg libvips`
- macOS: `brew install ffmpeg libvips`
- Windows:
  - Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Follow [libvips Windows installation guide](https://docs.moondream.ai/quick-start)

### Hardware Requirements

- GPU recommended for faster processing (CUDA compatible)
- Minimum 8GB RAM
- Storage space for temporary files and output videos

## Installation

1. Clone this repository and create a new virtual environment:

```bash
git clone https://github.com/vikhyat/moondream/blob/main/recipes/promptable-video-redaction
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install ffmpeg and libvips:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg libvips`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

> Downloading libvips for Windows requires some additional steps, see [here](https://docs.moondream.ai/quick-start)

## Usage

### Web Interface

1. Start the web interface:

```bash
python app.py
```

2. Open the provided URL in your browser (typically <http://localhost:7860>)

3. Use the interface to:
   - Upload your video file
   - Specify what to detect (e.g., "face", "logo", "text", "person wearing hat")
   - Choose visualization style (censor, bounding-box, hitmarker, SAM, or SAM-fast)
   - Configure advanced settings:
     - Processing speed/quality
     - Grid size for detection
     - Test mode for quick validation
   - Process the video and download results
   - Analyze detection patterns with visualization tools

### Command Line Interface

1. Create an `inputs` directory and place your videos there:

```bash
mkdir inputs
# Copy your videos to the inputs directory
```

2. Supported video formats:
   - .mp4
   - .avi
   - .mov
   - .mkv
   - .webm

3. Run the script with desired options:

```bash
python main.py [options]
```

### Command Line Options

- `--test`: Process only first 3 seconds (for testing settings)

```bash
python main.py --test
```

- `--preset`: Choose FFmpeg encoding preset (speed vs. quality)

```bash
python main.py --preset ultrafast  # Fastest, lower quality
python main.py --preset veryslow   # Slowest, highest quality
```

Available presets: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow

- `--detect`: Specify what to detect (using natural language)

```bash
python main.py --detect "person"  # Detect people
python main.py --detect "red car"  # Detect red cars
python main.py --detect "person wearing a hat"  # Detect people with hats
```

- `--box-style`: Choose visualization style

```bash
python main.py --box-style censor      # Black boxes (default)
python main.py --box-style bounding-box  # Boxes with labels
python main.py --box-style hitmarker   # COD-style markers
python main.py --box-style sam         # SAM segmentation
python main.py --box-style sam-fast    # Fast SAM segmentation
python main.py --box-style fuzzy-blur  # Gaussian blur effect
python main.py --box-style pixelated-blur  # Pixelation with blur
python main.py --box-style intense-pixelated-blur  # Strong pixelation
python main.py --box-style obfuscated-pixel  # Advanced pixelation
```

- `--rows` and `--cols`: Enable grid-based detection

```bash
python main.py --rows 2 --cols 2  # Split into 2x2 grid
python main.py --rows 3 --cols 3  # Split into 3x3 grid
```

Combine options as needed:

```bash
python main.py --detect "person wearing sunglasses" --box-style bounding-box --test --preset fast --rows 2 --cols 2
```

### Detection Analysis

The tool provides comprehensive visualization capabilities for analyzing detection patterns:

1. Real-time Video Visualization:
   - Frame-by-frame detection count
   - Current frame position indicator
   - Video statistics

2. Statistical Analysis:
   - Detections per frame over time
   - Detection area distribution
   - Average detection area trends
   - Detection center heatmap
   - Detection density timeline
   - Screen region analysis
   - Size-based categorization
   - Temporal pattern analysis

To analyze detection data:

```bash
python visualization.py path/to/detection_data.json
```

## Output Files

The tool generates two types of output files in the `outputs` directory:

1. Processed Videos:
   - Format: `[style]_[object_type]_[original_filename].mp4`
   - Examples:
     - `censor_face_video.mp4`
     - `bounding-box_person_video.mp4`
     - `hitmarker_car_video.mp4`
   - Encoding: H.264 codec for web compatibility
   - Quality: Configurable via FFmpeg presets

2. Detection Data:
   - Format: `[style]_[object_type]_[original_filename]_detections.json`
   - Contains frame-by-frame detection information
   - Used for visualization and analysis

## Detection Data Format

The detection data is saved in JSON format with the following structure:

```json
{
  "video_metadata": {
    "file_name": "example.mp4",
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "total_frames": 900,
    "duration_sec": 30,
    "detect_keyword": "face",
    "test_mode": false,
    "grid_size": "1x1",
    "box_style": "censor",
    "timestamp": "2024-03-14T12:00:00"
  },
  "frame_detections": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "objects": [
        {
          "keyword": "face",
          "bbox": [0.1, 0.1, 0.3, 0.3]  // [x1, y1, x2, y2] normalized coordinates
        }
      ]
    }
  ]
}
```

### Data Fields

1. Video Metadata:
   - `file_name`: Original video filename
   - `fps`: Frames per second
   - `width`, `height`: Video dimensions
   - `total_frames`: Total number of frames
   - `duration_sec`: Video duration in seconds
   - `detect_keyword`: Object type being detected
   - `test_mode`: Whether test mode was used
   - `grid_size`: Detection grid configuration
   - `box_style`: Visualization style used
   - `timestamp`: Processing timestamp

2. Frame Detections:
   - `frame`: Frame number (0-based)
   - `timestamp`: Frame timestamp in seconds
   - `objects`: List of detections in the frame
     - `keyword`: Object type detected
     - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
       - Coordinates are normalized (0.0 to 1.0)
       - (0,0) is top-left, (1,1) is bottom-right

## Notes

- Processing time depends on:
  - Video length and resolution
  - Grid size configuration
  - GPU availability and speed
  - Chosen visualization style (SAM is more intensive)
  - FFmpeg preset selection

- Best Practices:
  - Use test mode for initial configuration
  - Enable grid-based detection for crowded scenes
  - Choose visualization style based on use case:
    - Censor: Privacy and content moderation
    - Bounding Box: Analysis and debugging
    - Hitmarker: Stylistic visualization
    - SAM: Precise object segmentation
    - Fast SAM: Quick segmentation preview
  - Monitor system resources during processing
  - Use appropriate FFmpeg preset for your needs

- Known Limitations:
  - SAM visualization requires more processing power
  - Grid-based detection increases processing time
  - Test mode processes only first 3 seconds
  - Some visualization features require sufficient detection data

## Technical Details

### Scene Detection and Tracking

The tool uses advanced scene detection and object tracking:

1. Scene Detection:
   - Powered by PySceneDetect's ContentDetector
   - Automatically identifies scene changes in videos
   - Configurable detection threshold (default: 30.0)
   - Helps maintain tracking accuracy across scene boundaries

2. Object Tracking:
   - DeepSORT tracking for consistent object identification
   - Automatic tracker reset at scene changes
   - Maintains object identity within scenes
   - Prevents tracking errors across scene boundaries

3. Integration Benefits:
   - More accurate object tracking
   - Better handling of scene transitions
   - Reduced false positives in tracking
   - Improved tracking consistency

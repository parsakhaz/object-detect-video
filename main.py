#!/usr/bin/env python3
import cv2, os, subprocess, argparse
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

# Constants
FRAME_INTERVAL = 1  # Process every frame
BATCH_SIZE = 8      # Process 8 frames at a time
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
MARGIN = 10
PADDING = 10
MIN_LINE_HEIGHT = 20
TEST_MODE_DURATION = 3  # Process only first 3 seconds in test mode
FFMPEG_PRESETS = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']

# Detection parameters
IOU_THRESHOLD = 0.5  # IoU threshold for considering boxes related

def load_moondream():
    """Load Moondream model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        device_map={"": "cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
    return model, tokenizer

def get_video_properties(video_path):
    """Get basic video properties."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    return {'fps': fps, 'frame_count': frame_count, 'width': width, 'height': height}

def is_valid_box(box):
    """Check if box coordinates are reasonable."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Reject boxes that are too large (over 90% of frame in both dimensions)
    if width > 0.9 and height > 0.9:
        return False
        
    # Reject boxes that are too small (less than 1% of frame)
    if width < 0.01 or height < 0.01:
        return False
    
    return True

def split_frame_into_tiles(frame, rows, cols):
    """Split a frame into a grid of tiles."""
    height, width = frame.shape[:2]
    tile_height = height // rows
    tile_width = width // cols
    tiles = []
    tile_positions = []
    
    for i in range(rows):
        for j in range(cols):
            y1 = i * tile_height
            y2 = (i + 1) * tile_height if i < rows - 1 else height
            x1 = j * tile_width
            x2 = (j + 1) * tile_width if j < cols - 1 else width
            
            tile = frame[y1:y2, x1:x2]
            tiles.append(tile)
            tile_positions.append((x1, y1, x2, y2))
    
    return tiles, tile_positions

def convert_tile_coords_to_frame(box, tile_pos, frame_shape):
    """Convert coordinates from tile space to frame space."""
    frame_height, frame_width = frame_shape[:2]
    tile_x1, tile_y1, tile_x2, tile_y2 = tile_pos
    tile_width = tile_x2 - tile_x1
    tile_height = tile_y2 - tile_y1
    
    x1_tile_abs = box[0] * tile_width
    y1_tile_abs = box[1] * tile_height
    x2_tile_abs = box[2] * tile_width
    y2_tile_abs = box[3] * tile_height
    
    x1_frame_abs = tile_x1 + x1_tile_abs
    y1_frame_abs = tile_y1 + y1_tile_abs
    x2_frame_abs = tile_x1 + x2_tile_abs
    y2_frame_abs = tile_y1 + y2_tile_abs
    
    x1_norm = x1_frame_abs / frame_width
    y1_norm = y1_frame_abs / frame_height
    x2_norm = x2_frame_abs / frame_width
    y2_norm = y2_frame_abs / frame_height
    
    x1_norm = max(0.0, min(1.0, x1_norm))
    y1_norm = max(0.0, min(1.0, y1_norm))
    x2_norm = max(0.0, min(1.0, x2_norm))
    y2_norm = max(0.0, min(1.0, y2_norm))
    
    return [x1_norm, y1_norm, x2_norm, y2_norm]

def merge_tile_detections(tile_detections, iou_threshold=0.5):
    """Merge detections from different tiles using NMS-like approach."""
    if not tile_detections:
        return []
    
    all_boxes = []
    all_keywords = []
    
    # Collect all boxes and their keywords
    for detections in tile_detections:
        for box, keyword in detections:
            all_boxes.append(box)
            all_keywords.append(keyword)
    
    if not all_boxes:
        return []
    
    # Convert to numpy for easier processing
    boxes = np.array(all_boxes)
    
    # Calculate areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort boxes by area
    order = areas.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # Calculate IoU with rest of boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Get indices of boxes with IoU less than threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [(all_boxes[i], all_keywords[i]) for i in keep]

def detect_ads_in_frame(model, tokenizer, image, detect_keyword, rows=1, cols=1):
    """Detect objects in a frame using grid-based detection."""
    if rows == 1 and cols == 1:
        return detect_ads_in_frame_single(model, tokenizer, image, detect_keyword)
    
    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split frame into tiles
    tiles, tile_positions = split_frame_into_tiles(image, rows, cols)
    
    # Process each tile
    tile_detections = []
    for tile, tile_pos in zip(tiles, tile_positions):
        # Convert tile to PIL Image
        tile_pil = Image.fromarray(tile)
        
        # Detect objects in tile
        response = model.detect(tile_pil, detect_keyword)
        
        if response and "objects" in response and response["objects"]:
            objects = response["objects"]
            tile_objects = []
            
            for obj in objects:
                if all(k in obj for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                    box = [
                        obj['x_min'],
                        obj['y_min'],
                        obj['x_max'],
                        obj['y_max']
                    ]
                    
                    if is_valid_box(box):
                        # Convert tile coordinates to frame coordinates
                        frame_box = convert_tile_coords_to_frame(box, tile_pos, image.shape)
                        tile_objects.append((frame_box, detect_keyword))
            
            if tile_objects:  # Only append if we found valid objects
                tile_detections.append(tile_objects)
    
    # Merge detections from all tiles
    merged_detections = merge_tile_detections(tile_detections)
    return merged_detections

def detect_ads_in_frame_single(model, tokenizer, image, detect_keyword):
    """Single-frame detection function."""
    detected_objects = []
    
    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Detect objects
    response = model.detect(image, detect_keyword)
    print(f"\nDetection response for '{detect_keyword}':")
    print(response)
    
    # Check if we have valid objects
    if response and "objects" in response and response["objects"]:
        objects = response["objects"]
        print(f"Found {len(objects)} potential {detect_keyword} region(s)")
        
        for obj in objects:
            if all(k in obj for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                box = [
                    obj['x_min'],
                    obj['y_min'],
                    obj['x_max'],
                    obj['y_max']
                ]
                # If box is valid (not full-frame), add it
                if is_valid_box(box):
                    detected_objects.append((box, detect_keyword))
                    print(f"Added valid detection: {box}")
    
    return detected_objects

def draw_ad_boxes(frame, detected_objects, detect_keyword):
    """Draw black censor bars over detected objects."""
    height, width = frame.shape[:2]
    
    for (box, keyword) in detected_objects:
        try:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Only draw if box has reasonable size
            if x2 > x1 and y2 > y1:
                # Draw solid black rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                print(f"Drew censor bar at coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        except Exception as e:
            print(f"Error drawing censor bar {box}: {str(e)}")
    
    return frame

def filter_temporal_outliers(detections_dict, max_size_change=0.15):
    """Filter out extremely large detections that take up most of the frame.
    Only keeps detections that are reasonable in size.
    
    Args:
        detections_dict: Dictionary of {timestamp: [(box, keyword), ...]}
        max_size_change: Not used, kept for compatibility
    """
    filtered_detections = {}
    
    for t, detections in detections_dict.items():
        # Only keep detections that aren't too large
        valid_detections = []
        for box, keyword in detections:
            # Calculate box size as percentage of frame
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            
            # If box is less than 90% of frame, keep it
            if area < 0.9:
                valid_detections.append((box, keyword))
        
        if valid_detections:
            filtered_detections[t] = valid_detections
    
    return filtered_detections

def describe_frames(video_path, model, tokenizer, detect_keyword, test_mode=False, rows=1, cols=1):
    """Extract and detect objects in frames."""
    props = get_video_properties(video_path)
    fps = props['fps']
    
    # If in test mode, only process first 3 seconds
    if test_mode:
        frame_count = min(int(fps * TEST_MODE_DURATION), props['frame_count'])
    else:
        frame_count = props['frame_count']
    
    ad_detections = {}  # Store detection results by frame number
    
    print("Extracting frames and detecting objects...")
    video = cv2.VideoCapture(video_path)
    
    # Process every frame
    frame_count_processed = 0
    with tqdm(total=frame_count) as pbar:
        while frame_count_processed < frame_count:
            ret, frame = video.read()
            if not ret:
                break
            
            # Detect objects in the frame
            detected_objects = detect_ads_in_frame(model, tokenizer, frame, detect_keyword, rows=rows, cols=cols)
            
            # Store results for every frame, even if empty
            ad_detections[frame_count_processed] = detected_objects
            if detected_objects:
                print(f"\nFrame {frame_count_processed} detections:")
                for box, keyword in detected_objects:
                    print(f"- {keyword}: {box}")
            
            frame_count_processed += 1
            pbar.update(1)
    
    video.release()
    
    if frame_count_processed == 0:
        print("No frames could be read from video")
        return {}
    
    # Filter out only extremely large detections
    ad_detections = filter_temporal_outliers(ad_detections)
    return ad_detections

def create_detection_video(video_path, ad_detections, detect_keyword, output_path=None, ffmpeg_preset='medium', test_mode=False):
    """Create video with detection boxes."""
    if output_path is None:
        os.makedirs('outputs', exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join('outputs', f'detected_{base_name}_web.mp4')

    props = get_video_properties(video_path)
    fps, width, height = props['fps'], props['width'], props['height']
    
    # If in test mode, only process first few seconds
    if test_mode:
        frame_count = min(int(fps * TEST_MODE_DURATION), props['frame_count'])
    else:
        frame_count = props['frame_count']
    
    video = cv2.VideoCapture(video_path)
    
    # Use more efficient temporary codec
    temp_output = output_path.replace('_web.mp4', '_temp.mp4')
    out = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    print("Creating detection video...")
    frame_count_processed = 0
    
    with tqdm(total=frame_count) as pbar:
        while frame_count_processed < frame_count:
            ret, frame = video.read()
            if not ret:
                break
            
            # Get detections for this exact frame
            if frame_count_processed in ad_detections:
                current_detections = ad_detections[frame_count_processed]
                if current_detections:
                    print(f"Drawing detections for frame {frame_count_processed}")
                    frame = draw_ad_boxes(frame, current_detections, detect_keyword)
            
            out.write(frame)
            frame_count_processed += 1
            pbar.update(1)
    
    video.release()
    out.release()
    
    # Convert to web-compatible format more efficiently
    subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_output,
        '-c:v', 'libx264',
        '-preset', ffmpeg_preset,
        '-crf', '23',
        '-movflags', '+faststart',  # Better web playback
        '-loglevel', 'error',
        output_path
    ])
    
    os.remove(temp_output)  # Remove the temporary file
    return output_path

def process_video(video_path, detect_keyword, test_mode=False, ffmpeg_preset='medium', rows=1, cols=1):
    """Process a single video file."""
    print(f"\nProcessing: {video_path}")
    print(f"Looking for: {detect_keyword}")
    
    # Load model
    print("Loading Moondream model...")
    model, tokenizer = load_moondream()
    
    # Process video - detect objects
    ad_detections = describe_frames(video_path, model, tokenizer, detect_keyword, test_mode, rows, cols)
    
    # Create video with detection boxes
    output_path = create_detection_video(video_path, ad_detections, detect_keyword, ffmpeg_preset=ffmpeg_preset, test_mode=test_mode)
    print(f"\nOutput saved to: {output_path}")

def main():
    """Process all videos in the inputs directory."""
    parser = argparse.ArgumentParser(description='Detect objects in videos using Moondream2')
    parser.add_argument('--test', action='store_true', help='Process only first 3 seconds of each video')
    parser.add_argument('--preset', choices=FFMPEG_PRESETS, default='medium',
                      help='FFmpeg encoding preset (default: medium). Faster presets = lower quality')
    parser.add_argument('--detect', type=str, default='face',
                      help='Object to detect in the video (default: face, use --detect "thing to detect" to override)')
    parser.add_argument('--rows', type=int, default=1,
                      help='Number of rows to split each frame into (default: 1)')
    parser.add_argument('--cols', type=int, default=1,
                      help='Number of columns to split each frame into (default: 1)')
    args = parser.parse_args()
    
    input_dir = 'inputs'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    video_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
                   and not f.startswith('captioned_')]
    
    if not video_files:
        print("No video files found in 'inputs' directory")
        return
    
    print(f"Found {len(video_files)} videos to process")
    print(f"Will detect: {args.detect}")
    if args.test:
        print("Running in test mode - processing only first 3 seconds of each video")
    print(f"Using FFmpeg preset: {args.preset}")
    print(f"Grid size: {args.rows}x{args.cols}")
    
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        process_video(video_path, args.detect, test_mode=args.test, ffmpeg_preset=args.preset,
                     rows=args.rows, cols=args.cols)

if __name__ == "__main__":
    main()

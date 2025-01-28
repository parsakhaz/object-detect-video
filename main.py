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
EMA_ALPHA = 0.6     # Increased alpha for more immediate response
TEMPORAL_WINDOW = 3  # Reduced to only look at last 3 frames

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

def average_boxes(boxes):
    """Average box coordinates and remove outliers."""
    if not boxes:
        return None
        
    # Unzip boxes into separate coordinate lists
    coords = list(zip(*boxes))  # [[x1s], [y1s], [x2s], [y2s]]
    
    # Calculate mean and std for each coordinate
    means = [sum(coord_list) / len(coord_list) for coord_list in coords]
    
    # Calculate standard deviation for each coordinate
    stds = []
    for coord_list, mean in zip(coords, means):
        squared_diff = sum((x - mean) ** 2 for x in coord_list)
        std = (squared_diff / len(coord_list)) ** 0.5
        stds.append(std)
    
    # Filter out boxes with any coordinate more than 30% different from mean
    filtered_boxes = []
    for box in boxes:
        is_outlier = False
        for i, (coord, mean) in enumerate(zip(box, means)):
            if abs(coord - mean) > 0.3:  # 30% threshold
                is_outlier = True
                break
        if not is_outlier:
            filtered_boxes.append(box)
    
    if not filtered_boxes:
        return None
    
    # Calculate final average from filtered boxes
    final_coords = list(zip(*filtered_boxes))
    final_means = [sum(coord_list) / len(coord_list) for coord_list in final_coords]
    
    return final_means

def is_valid_box(box):
    """Check if box coordinates are reasonable (not covering entire frame)."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Only reject if the box covers almost the entire frame in BOTH dimensions
    # (i.e., full-screen detections)
    return not (width > 0.98 and height > 0.98)

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
    """Convert coordinates from tile space to frame space.
    
    Args:
        box: List [x1, y1, x2, y2] normalized coordinates in tile space (0-1)
        tile_pos: Tuple (x1, y1, x2, y2) absolute pixel coordinates of tile in frame
        frame_shape: Tuple (height, width) of original frame
    
    Returns:
        List [x1, y1, x2, y2] normalized coordinates in frame space (0-1)
    """
    # 1. Get tile and frame dimensions
    frame_height, frame_width = frame_shape[:2]
    tile_x1, tile_y1, tile_x2, tile_y2 = tile_pos
    tile_width = tile_x2 - tile_x1
    tile_height = tile_y2 - tile_y1
    
    # 2. Convert normalized tile coordinates (0-1) to absolute tile coordinates
    x1_tile_abs = box[0] * tile_width
    y1_tile_abs = box[1] * tile_height
    x2_tile_abs = box[2] * tile_width
    y2_tile_abs = box[3] * tile_height
    
    # 3. Convert absolute tile coordinates to absolute frame coordinates
    x1_frame_abs = tile_x1 + x1_tile_abs
    y1_frame_abs = tile_y1 + y1_tile_abs
    x2_frame_abs = tile_x1 + x2_tile_abs
    y2_frame_abs = tile_y1 + y2_tile_abs
    
    # 4. Normalize to frame coordinates (0-1)
    x1_norm = x1_frame_abs / frame_width
    y1_norm = y1_frame_abs / frame_height
    x2_norm = x2_frame_abs / frame_width
    y2_norm = y2_frame_abs / frame_height
    
    # 5. Ensure coordinates are within bounds
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

def detect_ads_in_frame(model, tokenizer, image, detect_keyword, previous_detections=None, rows=1, cols=1):
    """Detect objects in a frame using grid-based detection."""
    if rows == 1 and cols == 1:
        return detect_ads_in_frame_single(model, tokenizer, image, detect_keyword, previous_detections)
    
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
    
    if not merged_detections and previous_detections:
        return previous_detections
    
    return merged_detections

def detect_ads_in_frame_single(model, tokenizer, image, detect_keyword, previous_detections=None):
    """Original single-frame detection function."""
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
        
        has_valid_detection = False
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
                    has_valid_detection = True
                    print(f"Added valid detection: {box}")
        
        # If we only got full-frame detections and have previous detections, use those
        if not has_valid_detection and previous_detections:
            print("Using previous frame's detections due to full-frame results")
            return previous_detections
    elif previous_detections:
        # If no detections at all but we have previous detections, use those
        print("Using previous frame's detections due to no results")
        return previous_detections
    
    return detected_objects

def draw_ad_boxes(frame, detected_objects, detect_keyword):
    """Draw bounding boxes around detected objects."""
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
                # Draw red rectangle with thicker line
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Add label
                label = detect_keyword.capitalize()
                label_size = cv2.getTextSize(label, FONT, 0.7, 2)[0]
                cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0], y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1-6), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                print(f"Drew box at coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        except Exception as e:
            print(f"Error drawing box {box}: {str(e)}")
    
    return frame

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes intersect
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate areas
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def smooth_detections(current_frame, detections_history):
    """Apply temporal smoothing to detections using EMA and IoU matching."""
    if not detections_history:
        return current_frame
    
    # Get recent frames' detections - only look at last 2 frames for smoothing
    recent_detections = detections_history[-2:]  # More responsive
    
    smoothed_detections = []
    for current_box, keyword in current_frame:
        matched_boxes = []
        
        # Find related boxes in recent frames, with their recency index
        for i, frame_detections in enumerate(recent_detections):
            for past_box, _ in frame_detections:
                iou = calculate_iou(current_box, past_box)
                if iou > IOU_THRESHOLD:
                    # Store box with its recency weight (more recent = higher weight)
                    recency_weight = 0.8 if i == len(recent_detections)-1 else 0.2  # Much higher weight for most recent
                    matched_boxes.append((past_box, recency_weight))
                    break  # Only match one box per frame
        
        if matched_boxes:
            # Check if there's a significant position change from most recent detection
            if len(matched_boxes) >= 2:
                last_box = matched_boxes[-1][0]
                max_coord_change = max(abs(c - p) for c, p in zip(current_box, last_box))
                if max_coord_change > 0.05:  # More sensitive to changes (was 0.1)
                    # Big change detected - use current box with minimal smoothing
                    smoothed_box = list(current_box)
                    smoothed_detections.append((smoothed_box, keyword))
                    continue
            
            # Apply weighted EMA smoothing
            smoothed_box = list(current_box)
            total_weight = 0
            
            for past_box, recency_weight in matched_boxes:
                weight = recency_weight * EMA_ALPHA
                total_weight += weight
                for i in range(4):
                    smoothed_box[i] = smoothed_box[i] * (1 - weight) + past_box[i] * weight
            
            smoothed_detections.append((smoothed_box, keyword))
        else:
            # If no matches found, use current box as is
            smoothed_detections.append((current_box, keyword))
    
    return smoothed_detections

def describe_frames(video_path, model, tokenizer, detect_keyword, test_mode=False, rows=1, cols=1):
    """Extract and detect objects in frames."""
    props = get_video_properties(video_path)
    fps = props['fps']
    
    # If in test mode, only process first 3 seconds
    if test_mode:
        frame_count = min(int(fps * TEST_MODE_DURATION), props['frame_count'])
    else:
        frame_count = props['frame_count']
    
    ad_detections = {}  # Store detection results by timestamp
    detections_history = []  # Store recent detections for smoothing
    previous_detections = None  # Keep track of last valid detection
    
    print("Extracting frames and detecting objects...")
    video = cv2.VideoCapture(video_path)
    
    # Process every frame
    frame_count_processed = 0
    with tqdm(total=frame_count) as pbar:
        while frame_count_processed < frame_count:
            ret, frame = video.read()
            if not ret:
                break
            
            # Calculate exact timestamp for this frame
            timestamp = frame_count_processed / fps
            
            # Detect objects in the frame
            detected_objects = detect_ads_in_frame(model, tokenizer, frame, detect_keyword, 
                                                previous_detections, rows=rows, cols=cols)
            
            # Apply temporal smoothing
            if detected_objects:
                smoothed_objects = smooth_detections(detected_objects, detections_history)
                ad_detections[timestamp] = smoothed_objects
                detections_history.append(smoothed_objects)
                previous_detections = smoothed_objects
                print(f"\nFrame {frame_count_processed} (t={timestamp:.3f}s) detections:")
                for box, keyword in smoothed_objects:
                    print(f"- {keyword}: {box}")
            else:
                detections_history.append([])
            
            # Keep history window limited
            if len(detections_history) > TEMPORAL_WINDOW:
                detections_history.pop(0)
            
            frame_count_processed += 1
            pbar.update(1)
    
    video.release()
    
    if frame_count_processed == 0:
        print("No frames could be read from video")
        return {}
    
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
    
    # Get timestamps where we have detections
    detection_times = sorted(ad_detections.keys())
    print(f"Total frames with detections: {len(detection_times)}")
    
    print("Creating detection video...")
    frame_count_processed = 0
    
    with tqdm(total=frame_count) as pbar:
        while frame_count_processed < frame_count:
            ret, frame = video.read()
            if not ret:
                break
            
            # Calculate exact timestamp for this frame
            timestamp = frame_count_processed / fps
            
            # Find closest detection time
            current_detections = None
            for t in detection_times:
                if abs(t - timestamp) < 1/fps:  # Within one frame duration
                    current_detections = ad_detections[t]
                    break
            
            if current_detections:
                print(f"Drawing detections for frame {frame_count_processed} (t={timestamp:.3f}s)")
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

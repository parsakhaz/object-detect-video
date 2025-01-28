#!/usr/bin/env python3
import cv2, os, subprocess, argparse
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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
DETECT_KEYWORD = "advertisement"
IOU_THRESHOLD = 0.5  # IoU threshold for considering boxes related
EMA_ALPHA = 0.2     # Lower alpha for smoother transitions
TEMPORAL_WINDOW = 7  # Increased window for better smoothing
MIN_DETECTION_FRAMES = 3  # Minimum consecutive frames needed to confirm a detection
OUTLIER_THRESHOLD = 0.4  # Maximum allowed sudden change in box coordinates

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

def detect_ads_in_frame(model, tokenizer, image, previous_detections=None):
    """Detect marketing content in a frame."""
    detected_ads = []
    
    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Detect marketing objects
    response = model.detect(image, DETECT_KEYWORD)
    print(f"\nDetection response for '{DETECT_KEYWORD}':")
    print(response)
    
    # Check if we have valid objects
    if response and "objects" in response and response["objects"]:
        objects = response["objects"]
        print(f"Found {len(objects)} potential marketing region(s)")
        
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
                    detected_ads.append((box, DETECT_KEYWORD))
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
    
    return detected_ads

def draw_ad_boxes(frame, detected_ads):
    """Draw bounding boxes around detected marketing content."""
    height, width = frame.shape[:2]
    
    for (box, keyword) in detected_ads:
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
                label = "Advertisement"  # Use the actual keyword
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

def is_stable_detection(box, recent_detections):
    """Check if a detection is stable across multiple frames."""
    if not recent_detections:
        return False
        
    # Count how many recent frames have a similar box
    similar_boxes = 0
    for frame_detections in recent_detections:
        for past_box, _ in frame_detections:
            if calculate_iou(box, past_box) > IOU_THRESHOLD:
                similar_boxes += 1
                break
    
    # Require detection to be present in minimum number of recent frames
    return similar_boxes >= MIN_DETECTION_FRAMES

def is_smooth_transition(current_box, previous_box):
    """Check if the transition between boxes is smooth enough."""
    if not previous_box:
        return True
        
    # Check each coordinate for sudden changes
    for curr, prev in zip(current_box, previous_box):
        if abs(curr - prev) > OUTLIER_THRESHOLD:
            return False
    return True

def smooth_detections(current_frame, detections_history):
    """Apply temporal smoothing to detections using EMA and IoU matching."""
    if not detections_history:
        return current_frame
    
    # Get recent frames' detections
    recent_detections = detections_history[-TEMPORAL_WINDOW:]
    
    smoothed_detections = []
    for current_box, keyword in current_frame:
        # Only process if detection is stable across multiple frames
        if is_stable_detection(current_box, recent_detections):
            matched_boxes = []
            previous_smoothed = None
            
            # Find related boxes in recent frames
            for frame_detections in reversed(recent_detections):  # Process most recent first
                for past_box, _ in frame_detections:
                    if calculate_iou(current_box, past_box) > IOU_THRESHOLD:
                        matched_boxes.append(past_box)
                        if not previous_smoothed:
                            previous_smoothed = past_box
                        break
            
            if matched_boxes:
                # Check if transition is smooth
                smoothed_box = list(current_box)
                if previous_smoothed and is_smooth_transition(current_box, previous_smoothed):
                    # Apply EMA smoothing with more weight to history
                    for i in range(4):
                        smoothed_box[i] = (EMA_ALPHA * current_box[i] + 
                                         (1 - EMA_ALPHA) * previous_smoothed[i])
                    
                    smoothed_detections.append((smoothed_box, keyword))
                elif len(matched_boxes) >= MIN_DETECTION_FRAMES:
                    # If transition isn't smooth but detection is consistent, use average
                    avg_box = [sum(coord) / len(matched_boxes) for coord in zip(*matched_boxes)]
                    smoothed_detections.append((avg_box, keyword))
    
    return smoothed_detections

def describe_frames(video_path, model, tokenizer, test_mode=False):
    """Extract and detect ads in frames."""
    props = get_video_properties(video_path)
    
    # If in test mode, only process first 3 seconds
    if test_mode:
        frame_count = min(int(props['fps'] * TEST_MODE_DURATION), props['frame_count'])
    else:
        frame_count = props['frame_count']
    
    ad_detections = {}  # Store ad detection results by frame number
    detections_history = []  # Store recent detections for smoothing
    previous_detections = None  # Keep track of last valid detection
    
    print("Extracting frames and detecting ads...")
    video = cv2.VideoCapture(video_path)
    
    # Process every frame
    current_frame = 0
    with tqdm(total=frame_count) as pbar:
        while current_frame < frame_count:
            ret, frame = video.read()
            if not ret:
                break
            
            # Detect ads in the frame
            detected_ads = detect_ads_in_frame(model, tokenizer, frame, previous_detections)
            
            # Apply temporal smoothing
            if detected_ads:
                smoothed_ads = smooth_detections(detected_ads, detections_history)
                if smoothed_ads:  # Only store if we have stable detections
                    ad_detections[current_frame] = smoothed_ads
                    previous_detections = smoothed_ads
                    print(f"\nFrame {current_frame} detections:")
                    for box, keyword in smoothed_ads:
                        print(f"- {keyword}: {box}")
            
            # Always append to history, even if empty
            detections_history.append(detected_ads)
            
            # Keep history window limited
            if len(detections_history) > TEMPORAL_WINDOW:
                detections_history.pop(0)
            
            current_frame += 1
            pbar.update(1)
    
    video.release()
    
    if current_frame == 0:
        print("No frames could be read from video")
        return {}
    
    return ad_detections

def create_detection_video(video_path, ad_detections, output_path=None, ffmpeg_preset='medium'):
    """Create video with ad detection boxes only."""
    if output_path is None:
        os.makedirs('outputs', exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join('outputs', f'detected_{base_name}_web.mp4')

    props = get_video_properties(video_path)
    fps, width, height = props['fps'], props['width'], props['height']
    
    video = cv2.VideoCapture(video_path)
    
    # Use more efficient temporary codec
    temp_output = output_path.replace('_web.mp4', '_temp.mp4')
    out = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # Get the frame numbers where we have detections
    detection_frames = sorted(ad_detections.keys())
    print(f"Total frames with detections: {len(detection_frames)}")
    
    print("Creating detection video...")
    frame_count = 0
    
    with tqdm(total=props['frame_count']) as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Check if current frame has detections
            if frame_count in ad_detections:
                current_detections = ad_detections[frame_count]
                print(f"Drawing detections for frame {frame_count}: {current_detections}")
                frame = draw_ad_boxes(frame, current_detections)
            
            out.write(frame)
            frame_count += 1
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

def process_video(video_path, test_mode=False, ffmpeg_preset='medium'):
    """Process a single video file."""
    print(f"\nProcessing: {video_path}")
    
    # Load model
    print("Loading Moondream model...")
    model, tokenizer = load_moondream()
    
    # Process video - detect ads
    ad_detections = describe_frames(video_path, model, tokenizer, test_mode)
    
    # Create video with detection boxes
    output_path = create_detection_video(video_path, ad_detections, ffmpeg_preset=ffmpeg_preset)
    print(f"\nOutput saved to: {output_path}")

def main():
    """Process all videos in the inputs directory."""
    parser = argparse.ArgumentParser(description='Generate captions for videos using Moondream2')
    parser.add_argument('--test', action='store_true', help='Process only first 3 seconds of each video')
    parser.add_argument('--preset', choices=FFMPEG_PRESETS, default='medium',
                      help='FFmpeg encoding preset (default: medium). Faster presets = lower quality')
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
    if args.test:
        print("Running in test mode - processing only first 3 seconds of each video")
    print(f"Using FFmpeg preset: {args.preset}")
    
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        process_video(video_path, test_mode=args.test, ffmpeg_preset=args.preset)

if __name__ == "__main__":
    main()
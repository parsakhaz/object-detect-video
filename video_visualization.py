import os
import tempfile
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from persistence import load_detection_data

def create_frame_data(json_path):
    """Create frame-by-frame detection data for visualization."""
    data = load_detection_data(json_path)
    if not data:
        return None
    
    # Extract video metadata
    metadata = data["video_metadata"]
    fps = metadata["fps"]
    total_frames = metadata["total_frames"]
    
    # Create frame data
    frame_counts = {}
    for frame_data in data["frame_detections"]:
        frame_num = frame_data["frame"]
        frame_counts[frame_num] = len(frame_data["objects"])
    
    # Fill in missing frames with 0 detections
    for frame in range(total_frames):
        if frame not in frame_counts:
            frame_counts[frame] = 0
    
    # Convert to DataFrame
    df = pd.DataFrame(list(frame_counts.items()), columns=["frame", "detections"])
    df["timestamp"] = df["frame"] / fps
    
    return df, metadata

def generate_frame_image(df, frame_num, temp_dir, max_y):
    """Generate and save a single frame of the visualization."""
    plt.figure(figsize=(10, 6))
    
    # Plot data up to current frame
    current_data = df[df['frame'] <= frame_num]
    plt.plot(df['frame'], df['detections'], color='lightgray', alpha=0.5)  # Full data in background
    plt.plot(current_data['frame'], current_data['detections'], color='blue')
    
    # Add vertical line for current position
    plt.axvline(x=frame_num, color='red', linestyle='-', alpha=0.7)
    
    # Set consistent axes
    plt.xlim(0, len(df) - 1)
    plt.ylim(0, max_y * 1.1)  # Add 10% padding
    
    # Add labels
    plt.title(f'Frame {frame_num} - Detections Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Detections')
    
    # Add current stats
    current_detections = df[df['frame'] == frame_num]['detections'].iloc[0]
    plt.text(0.02, 0.98, f'Current detections: {current_detections}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    # Save frame
    frame_path = os.path.join(temp_dir, f'frame_{frame_num:05d}.png')
    plt.savefig(frame_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    return frame_path

def create_video_visualization(json_path):
    """Create a video visualization of the detection data."""
    try:
        # Load and process data
        frame_data, metadata = create_frame_data(json_path)
        if frame_data is None:
            return None, "No data found"
        
        total_frames = metadata["total_frames"]  # Use metadata frame count
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            max_y = frame_data['detections'].max()
            
            # Generate each frame
            print("Generating frames...")
            frame_paths = []
            with tqdm(total=total_frames, desc="Generating frames") as pbar:
                for frame in range(total_frames):  # Use total_frames instead of len(frame_data)
                    frame_path = generate_frame_image(frame_data, frame, temp_dir, max_y)
                    frame_paths.append(frame_path)
                    pbar.update(1)
            
            # Create output video path
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_video = os.path.join(output_dir, "detection_visualization.mp4")
            
            # Create temp output path
            base, ext = os.path.splitext(output_video)
            temp_output = f"{base}_temp{ext}"
            
            # First pass: Create video with OpenCV VideoWriter
            print("Creating initial video...")
            # Get frame size from first image
            first_frame = cv2.imread(frame_paths[0])
            height, width = first_frame.shape[:2]
            
            out = cv2.VideoWriter(
                temp_output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata["fps"],
                (width, height)
            )
            
            with tqdm(total=total_frames, desc="Creating video") as pbar:  # Use total_frames here too
                for frame_path in frame_paths:
                    frame = cv2.imread(frame_path)
                    out.write(frame)
                    pbar.update(1)
            
            out.release()
            
            # Second pass: Convert to web-compatible format
            print("Converting to web format...")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_output,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "23",
                        "-movflags",
                        "+faststart",  # Better web playback
                        "-loglevel",
                        "error",
                        output_video,
                    ],
                    check=True,
                )

                os.remove(temp_output)  # Remove the temporary file

                if not os.path.exists(output_video):
                    print(f"Warning: FFmpeg completed but output file not found at {output_video}")
                    return None, "Failed to create video"

                # Return video path and stats
                stats = f"""Video Stats:
FPS: {metadata['fps']}
Total Frames: {metadata['total_frames']}
Duration: {metadata['duration_sec']:.2f} seconds
Max Detections in a Frame: {frame_data['detections'].max()}
Average Detections per Frame: {frame_data['detections'].mean():.2f}"""
                
                return output_video, stats

            except subprocess.CalledProcessError as e:
                print(f"Error running FFmpeg: {str(e)}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                return None, f"Error creating visualization: {str(e)}"
        
    except Exception as e:
        print(f"Error creating video visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error creating visualization: {str(e)}" 
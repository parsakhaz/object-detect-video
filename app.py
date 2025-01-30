#!/usr/bin/env python3
import gradio as gr
import os
from main import load_moondream, process_video
import tempfile
import shutil

# Initialize model globally for reuse
print("Loading Moondream model...")
model, tokenizer = load_moondream()

def process_video_file(video_file, detect_keyword, ffmpeg_preset, rows, cols):
    """Process a video file through the Gradio interface."""
    try:
        # Create temporary directory for input
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up input directory
            input_dir = os.path.join(temp_dir, 'inputs')
            os.makedirs(input_dir, exist_ok=True)
            
            # Copy uploaded video to inputs directory
            input_video_path = os.path.join(input_dir, os.path.basename(video_file.name))
            shutil.copy2(video_file.name, input_video_path)
            
            # Process the video - output will be directly in outputs directory
            output_path = process_video(
                input_video_path,
                detect_keyword,
                test_mode=False,
                ffmpeg_preset=ffmpeg_preset,
                rows=rows,
                cols=cols
            )
            
            return output_path
            
    except Exception as e:
        raise gr.Error(f"Error processing video: {str(e)}")

# Create the Gradio interface
with gr.Blocks(title="Video Censoring App") as app:
    gr.Markdown("# Video Censoring App")
    gr.Markdown("Upload a video and specify what you want to censor. The app will process the video and return a censored version.")
    
    with gr.Row():
        with gr.Column():
            # Input components
            video_input = gr.Video(label="Upload Video")
            detect_input = gr.Textbox(label="What to Censor", placeholder="e.g. face, logo, text", value="face")
            preset_input = gr.Dropdown(
                choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                value='medium',
                label="Processing Speed (faster = lower quality)"
            )
            with gr.Row():
                rows_input = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Grid Rows")
                cols_input = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Grid Columns")
            
            process_btn = gr.Button("Process Video", variant="primary")
        
        with gr.Column():
            # Output components
            video_output = gr.Video(label="Censored Video")
            
    # Event handler
    process_btn.click(
        fn=process_video_file,
        inputs=[video_input, detect_input, preset_input, rows_input, cols_input],
        outputs=video_output
    )
    
    gr.Markdown("""
    ### Instructions
    1. Upload a video file
    2. Enter what you want to censor (e.g., 'face', 'logo', 'text')
    3. Adjust processing settings if needed:
        - Processing Speed: Faster presets process quicker but may reduce quality
        - Grid Size: Larger grids can help detect smaller objects but process slower, will have issues if a detection is split across multiple grid cells
    4. Click 'Process Video' and wait for the censored video
    """)

if __name__ == "__main__":
    app.launch(share=True) 
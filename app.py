#!/usr/bin/env python3
import gradio as gr
import os
from main import load_moondream, process_video
import tempfile
import shutil

# Initialize model globally for reuse
print("Loading Moondream model...")
model, tokenizer = load_moondream()

def process_video_file(video_file, detect_keyword, box_style, ffmpeg_preset, rows, cols):
    """Process a video file through the Gradio interface."""
    try:
        if not video_file:
            raise gr.Error("Please upload a video file")
            
        # Create temporary directory for input
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up input directory
            input_dir = os.path.join(temp_dir, 'inputs')
            os.makedirs(input_dir, exist_ok=True)
            
            # Get video filename and copy to inputs directory
            video_filename = os.path.basename(video_file)
            input_video_path = os.path.join(input_dir, video_filename)
            shutil.copy2(video_file, input_video_path)
            
            # Process the video - output will be directly in outputs directory
            output_path = process_video(
                input_video_path,
                detect_keyword,
                test_mode=False,
                ffmpeg_preset=ffmpeg_preset,
                rows=rows,
                cols=cols,
                box_style=box_style
            )
            
            # Copy output file to a temporary file that Gradio can access
            temp_output = os.path.join(temp_dir, 'output.mp4')
            shutil.copy2(output_path, temp_output)
            
            return temp_output
            
    except Exception as e:
        raise gr.Error(f"Error processing video: {str(e)}")

# Create the Gradio interface
with gr.Blocks(title="Video Object Detection with Moondream") as app:
    gr.Markdown("# Video Object Detection with Moondream")
    gr.Markdown("""
    This app uses [Moondream](https://github.com/vikhyat/moondream), a powerful yet lightweight vision-language model, 
    to detect and visualize objects in videos. Moondream can recognize a wide variety of objects, people, text, and more 
    with high accuracy while being much smaller than traditional models.
    
    Upload a video and specify what you want to detect. The app will process each frame using Moondream and visualize 
    the detections using your chosen style.
    """)
    
    with gr.Row():
        with gr.Column():
            # Input components
            video_input = gr.Video(label="Upload Video")
            detect_input = gr.Textbox(
                label="What to Detect", 
                placeholder="e.g. face, logo, text, person, car, dog, etc.", 
                value="face",
                info="Moondream can detect almost anything you can describe in natural language"
            )
            box_style_input = gr.Radio(
                choices=['censor', 'yolo', 'hitmarker'],
                value='censor',
                label="Visualization Style",
                info="Choose how to display detections"
            )
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
            video_output = gr.Video(label="Processed Video")
            
    # Event handler
    process_btn.click(
        fn=process_video_file,
        inputs=[video_input, detect_input, box_style_input, preset_input, rows_input, cols_input],
        outputs=video_output
    )
    
    gr.Markdown("""
    ### Instructions
    1. Upload a video file
    2. Enter what you want to detect (e.g., 'face', 'logo', 'text')
    3. Choose visualization style:
        - Censor: Black boxes over detected objects
        - YOLO: Traditional bounding boxes with labels
        - Hitmarker: Call of Duty style crosshair markers
    4. Adjust processing settings if needed:
        - Processing Speed: Faster presets process quicker but may reduce quality
        - Grid Size: Larger grids can help detect smaller objects but process slower
    5. Click 'Process Video' and wait for the result
    
    ### About Moondream
    Moondream is a tiny yet powerful vision-language model that can analyze images and answer questions about them. 
    It's designed to be lightweight and efficient while maintaining high accuracy. Some key features:
    - Only 2B parameters (compared to 80B+ in other models)
    - Fast inference with minimal resource requirements
    - Supports CPU and GPU execution
    - Open source and free to use
    
    Links:
    - [GitHub Repository](https://github.com/vikhyat/moondream)
    - [Hugging Face Space](https://huggingface.co/vikhyatk/moondream2)
    - [Python Package](https://pypi.org/project/moondream/)
    """)

if __name__ == "__main__":
    app.launch(share=True)
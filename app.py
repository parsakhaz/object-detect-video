#!/usr/bin/env python3
import gradio as gr
import os
from main import load_moondream, process_video, load_sam_model
import tempfile
import shutil
import torch
from visualization import visualize_detections
from persistence import load_detection_data
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import pandas as pd

# import spaces

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Check CUDA availability
print(f"Is CUDA available: {torch.cuda.is_available()}")
# We want to get True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# GPU Name

# Initialize models globally for reuse
print("Loading Moondream model...")
model, tokenizer = load_moondream()
print("Loading SAM model...")
sam_model, sam_processor = load_sam_model()


# Uncomment for Hugging Face Spaces
# @spaces.GPU(duration=120)
def process_video_file(
    video_file, detect_keyword, box_style, ffmpeg_preset, rows, cols, test_mode
):
    """Process a video file through the Gradio interface."""
    try:
        if not video_file:
            raise gr.Error("Please upload a video file")

        # Ensure input/output directories exist using absolute paths
        inputs_dir = os.path.join(WORKSPACE_ROOT, "inputs")
        outputs_dir = os.path.join(WORKSPACE_ROOT, "outputs")
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

        # Copy uploaded video to inputs directory
        video_filename = f"input_{os.path.basename(video_file)}"
        input_video_path = os.path.join(inputs_dir, video_filename)
        shutil.copy2(video_file, input_video_path)

        try:
            # Process the video
            output_path = process_video(
                input_video_path,
                detect_keyword,
                test_mode=test_mode,
                ffmpeg_preset=ffmpeg_preset,
                rows=rows,
                cols=cols,
                box_style=box_style,
            )

            # Get the corresponding JSON path
            base_name = os.path.splitext(os.path.basename(video_filename))[0]
            json_path = os.path.join(outputs_dir, f"{box_style}_{detect_keyword}_{base_name}_detections.json")

            # Verify output exists and is readable
            if not output_path or not os.path.exists(output_path):
                print(f"Warning: Output path {output_path} does not exist")
                # Try to find the output based on expected naming convention
                expected_output = os.path.join(
                    outputs_dir, f"{box_style}_{detect_keyword}_{video_filename}"
                )
                if os.path.exists(expected_output):
                    output_path = expected_output
                else:
                    # Try searching in outputs directory for any matching file
                    matching_files = [
                        f
                        for f in os.listdir(outputs_dir)
                        if f.startswith(f"{box_style}_{detect_keyword}_")
                    ]
                    if matching_files:
                        output_path = os.path.join(outputs_dir, matching_files[0])
                    else:
                        raise gr.Error("Failed to locate output video")

            # Convert output path to absolute path if it isn't already
            if not os.path.isabs(output_path):
                output_path = os.path.join(WORKSPACE_ROOT, output_path)

            print(f"Returning output path: {output_path}")
            return output_path, json_path

        finally:
            # Clean up input file
            try:
                if os.path.exists(input_video_path):
                    os.remove(input_video_path)
            except:
                pass

    except Exception as e:
        print(f"Error in process_video_file: {str(e)}")
        raise gr.Error(f"Error processing video: {str(e)}")

def create_visualization_plots(json_path):
    """Create visualization plots and return them as images."""
    try:
        # Load the data
        data = load_detection_data(json_path)
        if not data:
            return None, None, None, None, "No data found"

        # Convert to DataFrame
        rows = []
        for frame_data in data["frame_detections"]:
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            for obj in frame_data["objects"]:
                rows.append({
                    "frame": frame,
                    "timestamp": timestamp,
                    "keyword": obj["keyword"],
                    "x1": obj["bbox"][0],
                    "y1": obj["bbox"][1],
                    "x2": obj["bbox"][2],
                    "y2": obj["bbox"][3],
                    "area": (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1])
                })

        if not rows:
            return None, None, None, None, "No detections found in the data"

        df = pd.DataFrame(rows)
        plots = []

        # Create each plot and convert to image
        for plot_num in range(4):
            plt.figure(figsize=(8, 6))
            
            if plot_num == 0:
                # Plot 1: Number of detections per frame
                detections_per_frame = df.groupby("frame").size()
                plt.plot(detections_per_frame.index, detections_per_frame.values)
                plt.xlabel("Frame")
                plt.ylabel("Number of Detections")
                plt.title("Detections Per Frame")
            
            elif plot_num == 1:
                # Plot 2: Distribution of detection areas
                df["area"].hist(bins=30)
                plt.xlabel("Detection Area (normalized)")
                plt.ylabel("Count")
                plt.title("Distribution of Detection Areas")
            
            elif plot_num == 2:
                # Plot 3: Average detection area over time
                avg_area = df.groupby("frame")["area"].mean()
                plt.plot(avg_area.index, avg_area.values)
                plt.xlabel("Frame")
                plt.ylabel("Average Detection Area")
                plt.title("Average Detection Area Over Time")
            
            elif plot_num == 3:
                # Plot 4: Heatmap of detection centers
                df["center_x"] = (df["x1"] + df["x2"]) / 2
                df["center_y"] = (df["y1"] + df["y2"]) / 2
                plt.hist2d(df["center_x"], df["center_y"], bins=30)
                plt.colorbar()
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                plt.title("Detection Center Heatmap")

            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plots.append(Image.open(buf))
            plt.close()

        # Generate summary text
        summary = f"""Summary Statistics:
Total frames analyzed: {len(data['frame_detections'])}
Total detections: {len(df)}
Average detections per frame: {len(df) / len(data['frame_detections']):.2f}

Video metadata:
"""
        for key, value in data["video_metadata"].items():
            summary += f"{key}: {value}\n"

        return plots[0], plots[1], plots[2], plots[3], summary

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"Error creating visualization: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Promptable Video Redaction") as app:
    with gr.Tabs():
        with gr.Tab("Process Video"):
            gr.Markdown("# Promptable Video Redaction with Moondream")
            gr.Markdown(
                """
            [Moondream 2B](https://github.com/vikhyat/moondream) is a lightweight vision model that detects and visualizes objects in videos. It can identify objects, people, text and more.

            Upload a video and specify what to detect. The app will process each frame and apply your chosen visualization style. For help, join the [Moondream Discord](https://discord.com/invite/tRUdpjDQfH).
            """
            )

            with gr.Row():
                with gr.Column():
                    # Input components
                    video_input = gr.Video(label="Upload Video")

                    detect_input = gr.Textbox(
                        label="What to Detect",
                        placeholder="e.g. face, logo, text, person, car, dog, etc.",
                        value="face",
                        info="Moondream can detect anything that you can describe in natural language",
                    )

                    gr.Examples(
                        examples=[
                            ["examples/homealone.mp4", "face"],
                            ["examples/soccer.mp4", "ball"],
                            ["examples/rally.mp4", "license plate"],
                        ],
                        inputs=[video_input, detect_input],
                        label="Try these examples",
                    )

                    process_btn = gr.Button("Process Video", variant="primary")

                    with gr.Accordion("Advanced Settings", open=False):
                        box_style_input = gr.Radio(
                            choices=["censor", "bounding-box", "hitmarker", "sam", "sam-fast"],
                            value="censor",
                            label="Visualization Style",
                            info="Choose how to display detections: censor (black boxes), bounding-box (red boxes with labels), hitmarker (COD-style markers), sam (precise segmentation), or sam-fast (faster but less precise segmentation)",
                        )
                        preset_input = gr.Dropdown(
                            choices=[
                                "ultrafast",
                                "superfast",
                                "veryfast",
                                "faster",
                                "fast",
                                "medium",
                                "slow",
                                "slower",
                                "veryslow",
                            ],
                            value="medium",
                            label="Processing Speed (faster = lower quality)",
                        )
                        with gr.Row():
                            rows_input = gr.Slider(
                                minimum=1, maximum=4, value=1, step=1, label="Grid Rows"
                            )
                            cols_input = gr.Slider(
                                minimum=1, maximum=4, value=1, step=1, label="Grid Columns"
                            )

                        test_mode_input = gr.Checkbox(
                            label="Test Mode (Process first 3 seconds only)",
                            value=True,
                            info="Enable to quickly test settings on a short clip before processing the full video (recommended)",
                        )

                        gr.Markdown(
                            """
                        Note: Processing in test mode will only process the first 3 seconds of the video and is recommended for testing settings.
                        """
                        )

                        gr.Markdown(
                            """
                        We can get a rough estimate of how long the video will take to process by multiplying the videos framerate * seconds * the number of rows and columns and assuming 0.12 seconds processing time per detection.
                        For example, a 3 second video at 30fps with 2x2 grid, the estimated time is 3 * 30 * 2 * 2 * 0.12 = 43.2 seconds (tested on a 4090 GPU).
                        
                        Note: Using the SAM visualization style will increase processing time significantly as it performs additional segmentation for each detection. The sam-fast option uses a smaller model for faster processing at the cost of some accuracy.
                        """
                        )

                with gr.Column():
                    # Output components
                    video_output = gr.Video(label="Processed Video")
                    json_output = gr.Text(label="Detection Data Path", visible=False)

                    # About section under the video output
                    gr.Markdown(
                        """
                    ### Links:
                    - [GitHub Repository](https://github.com/vikhyat/moondream)
                    - [Hugging Face](https://huggingface.co/vikhyatk/moondream2)
                    - [Python Package](https://pypi.org/project/moondream/)
                    - [Moondream Recipes](https://docs.moondream.ai/recipes)
                    """
                    )

        with gr.Tab("Analyze Results"):
            gr.Markdown("# Detection Analysis")
            gr.Markdown(
                """
            Analyze the detection results from processed videos. The analysis includes:
            - Number of detections per frame
            - Distribution of detection areas
            - Average detection area over time
            - Spatial distribution of detections
            """
            )
            
            with gr.Row():
                json_input = gr.File(
                    label="Upload Detection Data (JSON)",
                    file_types=[".json"],
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                with gr.Column():
                    plot1 = gr.Image(label="Detections Per Frame")
                    plot2 = gr.Image(label="Detection Areas Distribution")
                
                with gr.Column():
                    plot3 = gr.Image(label="Average Detection Area Over Time")
                    plot4 = gr.Image(label="Detection Centers Heatmap")
            
            stats_output = gr.Textbox(
                label="Statistics",
                lines=10,
                max_lines=15,
                interactive=False
            )

    # Event handlers
    process_outputs = process_btn.click(
        fn=process_video_file,
        inputs=[
            video_input,
            detect_input,
            box_style_input,
            preset_input,
            rows_input,
            cols_input,
            test_mode_input,
        ],
        outputs=[video_output, json_output],
    )

    # Auto-analyze after processing
    process_outputs.then(
        fn=create_visualization_plots,
        inputs=[json_output],
        outputs=[plot1, plot2, plot3, plot4, stats_output],
    )

    # Manual analysis button
    analyze_btn.click(
        fn=create_visualization_plots,
        inputs=[json_input],
        outputs=[plot1, plot2, plot3, plot4, stats_output],
    )

if __name__ == "__main__":
    app.launch(share=True)

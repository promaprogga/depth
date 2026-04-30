import gradio as gr
import cv2
import tempfile
import torch
import numpy as np
import os
import time
from depth_anything_3.api import DepthAnything3

def process_video(video_path):
    if not video_path:
        return None
        
    start_time = time.time()
    print(f"Processing video: {video_path}")
    
    # Load model on demand or reuse global one
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Using DA3-LARGE-1.1 or DA3MONO-LARGE for simple monocular depth
    # DA3MONO-LARGE is optimized for single-image/frame relative depth estimation
    model_name = "depth-anything/DA3MONO-LARGE"
    print(f"Loading model {model_name}...")
    model = DepthAnything3.from_pretrained(model_name).to(device)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    if not frames:
        return None
        
    print(f"Extracted {len(frames)} frames. Running inference...")
    
    out_frames = []
    
    from moviepy.editor import ImageSequenceClip
    
    # Use current directory for output for easier access
    out_path = os.path.join(os.getcwd(), "output_depth_side_by_side.mp4")
    
    h, w, _ = frames[0].shape
    processed_frames = []

    print("Processing frames...")
    # Process sequentially to avoid OOM
    for i in range(len(frames)):
        with torch.no_grad():
            pred = model.inference([frames[i]])
            depth_map = pred.depth[0] # [H, W] float
            
            # Normalize to 0-255 uint8 for video
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_norm = 255.0 * (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth_map)
                
            depth_uint8 = depth_norm.astype(np.uint8)
            
            # Apply color map
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
            
            # Resize depth map back to original frame size
            depth_color_resized = cv2.resize(depth_color, (w, h))
            
            # depth_color_resized is BGR, original frames[i] is RGB
            # Convert depth to RGB for moviepy
            depth_rgb = cv2.cvtColor(depth_color_resized, cv2.COLOR_BGR2RGB)
            
            # Stack side-by-side
            combined_frame = np.hstack([frames[i], depth_rgb])
            processed_frames.append(combined_frame)
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
            
    # Clear model from memory before video encoding to save VRAM
    del model
    torch.cuda.empty_cache()
    
    print("Encoding video with moviepy...")
    try:
        clip = ImageSequenceClip(processed_frames, fps=fps)
        clip.write_videofile(out_path, codec="libx264", audio=False)
    except Exception as e:
        error_msg = f"Error during video encoding: {str(e)}"
        print(error_msg)
        return None, error_msg
    
    end_time = time.time()
    duration = end_time - start_time
    success_msg = f"Processing completed in {duration:.2f} seconds"
    print(f"Finished. Output saved to {out_path}")
    
    return out_path, success_msg

with gr.Blocks(title="Simple DA3 Video Depth") as demo:
    gr.Markdown("# 🌊 Depth Anything 3: Simple Video Depth Estimation")
    gr.Markdown("Upload a video and get the corresponding depth map video.")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            btn = gr.Button("Process Video", variant="primary")
        with gr.Column():
            output_video = gr.Video(label="Side-by-Side Depth Video", interactive=False)
            output_text = gr.Textbox(label="Processing Stats", interactive=False)
            
    btn.click(fn=process_video, inputs=input_video, outputs=[output_video, output_text])

if __name__ == "__main__":
    # Launch app
    # Launch app - letting Gradio find an available port automatically
    demo.launch(server_name="0.0.0.0", share=False)

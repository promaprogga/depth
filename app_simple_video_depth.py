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
    
    # Use current directory for output for easier access
    out_path = os.path.join(os.getcwd(), "output_depth_side_by_side.mp4")
    
    h, w, _ = frames[0].shape
    # Side by side means 2 * width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))
    
    if not out_video.isOpened():
        print(f"Error: Could not open VideoWriter for {out_path}")
        return None

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
            
            # CRITICAL: Resize depth map back to original frame size to match input for side-by-side
            # and to match VideoWriter initialization
            depth_color_resized = cv2.resize(depth_color, (w, h))
            
            # Convert original frame back to BGR for OpenCV VideoWriter
            orig_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            
            # Stack side-by-side
            combined_frame = np.hstack([orig_bgr, depth_color_resized])
            
            out_video.write(combined_frame)
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
            
    out_video.release()
    
    # Clear memory
    del model
    torch.cuda.empty_cache()
    
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished processing in {duration:.2f} seconds. Output saved to {out_path}")
    return out_path, f"Processing completed in {duration:.2f} seconds"

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

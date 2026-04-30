import gradio as gr
import cv2
import tempfile
import torch
import numpy as np
import os
from depth_anything_3.api import DepthAnything3

def process_video(video_path):
    if not video_path:
        return None
        
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
    
    # Create temp output file
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    # Process sequentially to avoid OOM
    # For large videos, we do frame-by-frame
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
            
            # Apply color map (INFERNO is commonly used for depth)
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
            
            out_video.write(depth_color)
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
            
    out_video.release()
    
    # Clear memory
    del model
    torch.cuda.empty_cache()
    
    print(f"Finished processing. Output saved to {out_path}")
    return out_path

with gr.Blocks(title="Simple DA3 Video Depth") as demo:
    gr.Markdown("# 🌊 Depth Anything 3: Simple Video Depth Estimation")
    gr.Markdown("Upload a video and get the corresponding depth map video.")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            btn = gr.Button("Process Video", variant="primary")
        with gr.Column():
            output_video = gr.Video(label="Depth Video", interactive=False)
            
    btn.click(fn=process_video, inputs=input_video, outputs=output_video)

if __name__ == "__main__":
    # Launch app
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)

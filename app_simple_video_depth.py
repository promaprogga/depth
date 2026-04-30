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

    print("Processing frames and analyzing liveness...")
    # Process sequentially to avoid OOM
    liveness_scores = []
    
    for i in range(len(frames)):
        with torch.no_grad():
            pred = model.inference([frames[i]])
            depth_map = pred.depth[0] # [H, W] float
            
            # 1. Liveness Analysis: Calculate depth variance in the central region
            # Real faces have depth contours; screens/photos are flat.
            center_h, center_w = depth_map.shape
            # Analyze the middle 50% of the frame
            h_start, h_end = int(center_h * 0.25), int(center_h * 0.75)
            w_start, w_end = int(center_w * 0.25), int(center_w * 0.75)
            face_region = depth_map[h_start:h_end, w_start:w_end]
            
            # Normalize for consistent variance calculation
            d_min, d_max = face_region.min(), face_region.max()
            if d_max > d_min:
                norm_region = (face_region - d_min) / (d_max - d_min)
                # High variance (0.1+) indicates 3D structure; low variance indicates flatness
                variance = norm_region.std()
                liveness_scores.append(variance)
            else:
                liveness_scores.append(0)

            # 2. Visualization
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_norm = 255.0 * (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth_map)
                
            depth_uint8 = depth_norm.astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
            depth_color_resized = cv2.resize(depth_color, (w, h))
            depth_rgb = cv2.cvtColor(depth_color_resized, cv2.COLOR_BGR2RGB)
            
            combined_frame = np.hstack([frames[i], depth_rgb])
            processed_frames.append(combined_frame)
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
            
    # Clear model from memory
    del model
    torch.cuda.empty_cache()
    
    # Calculate average liveness score
    # A score of > 0.15 is generally a real 3D object; < 0.08 is likely a flat surface (spoof)
    avg_variance = np.mean(liveness_scores)
    liveness_conf = min(100, max(0, (avg_variance - 0.05) / 0.15 * 100))
    liveness_result = "REAL 3D" if liveness_conf > 50 else "FLAT/SPOOF"
    
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
    stats_msg = (
        f"Processing completed in {duration:.2f}s\n"
        f"Analysis: {liveness_result} (3Dness Score: {liveness_conf:.1f}%)\n"
        f"Note: High scores indicate 3D contours (Real), low scores indicate flat surfaces (AI/Spoof)."
    )
    print(f"Finished. {stats_msg}")
    
    return out_path, stats_msg

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

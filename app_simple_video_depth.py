import gradio as gr
import cv2
import tempfile
import torch
import numpy as np
import os
import time
from PIL import Image
from depth_anything_3.api import DepthAnything3

def process_media(file_path):
    if not file_path:
        return None, None, gr.update(visible=False), gr.update(visible=False), "Please upload a file."
        
    start_time = time.time()
    # File path can be a string or a file-like object depending on Gradio version/config
    filename = file_path.name if hasattr(file_path, "name") else file_path
    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    if not is_image and not is_video:
        return None, None, gr.update(visible=False), gr.update(visible=False), f"Unsupported format: {ext}"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "depth-anything/DA3MONO-LARGE"
    print(f"Loading model {model_name} on {device}...")
    model = DepthAnything3.from_pretrained(model_name).to(device)

    if is_image:
        print(f"Processing image: {filename}")
        img = cv2.imread(filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            pred = model.inference([img_rgb])
            depth_map = pred.depth[0]
            
        # Analysis
        center_h, center_w = depth_map.shape
        h_start, h_end = int(center_h * 0.25), int(center_h * 0.75)
        w_start, w_end = int(center_w * 0.25), int(center_w * 0.75)
        face_region = depth_map[h_start:h_end, w_start:w_end]
        d_min, d_max = face_region.min(), face_region.max()
        variance = face_region.std() / (d_max - d_min) if d_max > d_min else 0
        liveness_conf = min(100, max(0, (variance - 0.05) / 0.15 * 100))
        liveness_result = "REAL 3D" if liveness_conf > 50 else "FLAT/SPOOF"
        
        # Color mapping
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_norm = 255.0 * (depth_map - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depth_map)
        depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_rgb_final = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        
        duration = time.time() - start_time
        stats = f"Processed in {duration:.2f}s\nAnalysis: {liveness_result} (3Dness: {liveness_conf:.1f}%)"
        
        del model
        torch.cuda.empty_cache()
        
        return depth_rgb_final, None, gr.update(visible=True), gr.update(visible=False), stats

    else:
        from moviepy.editor import ImageSequenceClip
        print(f"Processing video: {filename}")
        
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 30.0
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames: return None, None, gr.update(visible=False), gr.update(visible=False), "Failed to read video"

        processed_frames = []
        liveness_scores = []
        h, w, _ = frames[0].shape

        print(f"Processing {len(frames)} frames...")
        for i in range(len(frames)):
            with torch.no_grad():
                pred = model.inference([frames[i]])
                depth_map = pred.depth[0]
                
                # Analysis
                h_start, h_end = int(depth_map.shape[0] * 0.25), int(depth_map.shape[0] * 0.75)
                w_start, w_end = int(depth_map.shape[1] * 0.25), int(depth_map.shape[1] * 0.75)
                face_region = depth_map[h_start:h_end, w_start:w_end]
                d_min, d_max = face_region.min(), face_region.max()
                variance = face_region.std() / (d_max - d_min) if d_max > d_min else 0
                liveness_scores.append(variance)
                
                # Visualization (Depth Only)
                d_min_full, d_max_full = depth_map.min(), depth_map.max()
                depth_norm = 255.0 * (depth_map - d_min_full) / (depth_max_full - d_min_full) if d_max_full > d_min_full else np.zeros_like(depth_map)
                depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
                depth_color_resized = cv2.resize(depth_color, (w, h))
                processed_frames.append(cv2.cvtColor(depth_color_resized, cv2.COLOR_BGR2RGB))
            
            if (i + 1) % 10 == 0: print(f"Processed {i + 1}/{len(frames)} frames")

        del model
        torch.cuda.empty_cache()

        avg_variance = np.mean(liveness_scores)
        liveness_conf = min(100, max(0, (avg_variance - 0.05) / 0.15 * 100))
        liveness_result = "REAL 3D" if liveness_conf > 50 else "FLAT/SPOOF"
        
        out_path = os.path.join(os.getcwd(), "output_depth.mp4")
        clip = ImageSequenceClip(processed_frames, fps=fps)
        clip.write_videofile(out_path, codec="libx264", audio=False)
        
        duration = time.time() - start_time
        stats = f"Processed in {duration:.2f}s\nInfo: {len(frames)} frames @ {fps:.1f} FPS\nAnalysis: {liveness_result} (3Dness: {liveness_conf:.1f}%)"
        
        return None, out_path, gr.update(visible=False), gr.update(visible=True), stats

with gr.Blocks(title="DA3 Media Depth") as demo:
    gr.Markdown("# 🌊 Depth Anything 3: Media Depth Estimation")
    gr.Markdown("Upload an **Image** or **Video** to generate a depth map.")
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Upload Image or Video")
            btn = gr.Button("Process Media", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Depth Image", visible=False)
            output_video = gr.Video(label="Depth Video", visible=False)
            output_text = gr.Textbox(label="Processing Stats", interactive=False)
            
    btn.click(
        fn=process_media, 
        inputs=input_file, 
        outputs=[output_image, output_video, output_image, output_video, output_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)

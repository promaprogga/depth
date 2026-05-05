#!/usr/bin/env python
"""CLI version of video depth estimation - processes video directly without Gradio UI."""

import argparse
import cv2
import torch
import numpy as np
import sys
import imageio
from depth_anything_3.api import DepthAnything3


def process_video(video_path, output_path=None):
    if not video_path:
        print("Error: No video path provided")
        return None
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (will be slow)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Using DA3MONO-LARGE for monocular depth estimation
    model_name = "depth-anything/DA3MONO-LARGE"
    print(f"Loading model {model_name}...")
    model = DepthAnything3.from_pretrained(model_name).to(device)
    print("Model loaded successfully")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    
    print(f"Input video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = video_path.replace('.mp4', '_depth.mp4').replace('.avi', '_depth.avi')
    
    # Use imageio for writing (more reliable than OpenCV VideoWriter)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    print(f"Output video: {output_path}")
    print("Processing frames...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        with torch.no_grad():
            pred = model.inference([frame_rgb])
            depth_map = pred.depth[0]  # [H, W] float
        
        # Normalize to 0-255 uint8
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_norm = 255.0 * (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_map)
        
        depth_uint8 = depth_norm.astype(np.uint8)
        
        # Apply color map (INFERNO)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        
        # Convert BGR to RGB for imageio
        depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        writer.append_data(depth_rgb)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
    
    cap.release()
    writer.close()
    
    # Clear memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Finished! Output saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Depth Anything 3 - Video Depth Estimation")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Path to output video file (default: input_depth.mp4)")
    
    args = parser.parse_args()
    
    result = process_video(args.input_video, args.output)
    
    if result is None:
        sys.exit(1)
    
    print(f"\nSuccess! Depth video saved to: {result}")


if __name__ == "__main__":
    main()
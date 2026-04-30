#!/usr/bin/env python
"""CLI version of video depth estimation - processes video directly without Gradio UI."""

import argparse
import cv2
import torch
import numpy as np
import sys
import os
from depth_anything_3.api import DepthAnything3


def process_video(video_path, output_path=None):
    if not video_path:
        print("Error: No video path provided")
        return None
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Using DA3MONO-LARGE for monocular depth estimation
    model_name = "depth-anything/DA3MONO-LARGE"
    print(f"Loading model {model_name}...")
    try:
        model = DepthAnything3.from_pretrained(model_name).to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
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
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_depth{ext}"
    
    # Use cv2.VideoWriter (matches app_simple_video_depth.py fix)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"Error: Cannot open video writer for: {output_path}")
        cap.release()
        return None

    print(f"Output video: {output_path}")
    print("Processing frames...")
    
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for model inference
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
            
            # Write to video (cv2 expects BGR)
            writer.write(depth_color)
            
            frame_count += 1
            if frame_count % 10 == 0:
                progress = (100 * frame_count / total_frames) if total_frames > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        cap.release()
        writer.release()
    
    # Clear memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Finished! Output saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Depth Anything 3 - Video Depth Estimation (CLI)")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Path to output video file (optional)")
    
    args = parser.parse_args()
    
    result = process_video(args.input_video, args.output)
    
    if result is None:
        sys.exit(1)
    
    print(f"\nSuccess! Depth video saved to: {result}")


if __name__ == "__main__":
    main()
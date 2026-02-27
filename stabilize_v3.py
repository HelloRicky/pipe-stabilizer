#!/usr/bin/env python3
"""
Pipe Camera Stabilizer v3 - Gravity Detection with Canvas Expansion
Keeps water at the bottom by rotating. Expands canvas to show full frame.
"""

import cv2
import numpy as np
import subprocess
import sys

def find_water_angle(frame):
    """
    Find which direction the water/dark region is.
    Returns angle where water is (0=right, 90=bottom, 180=left, -90=top)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    
    h, w = gray.shape
    center_x, center_y = w // 2, h // 2
    
    # Sample brightness in 36 directions (every 10 degrees)
    radius = min(h, w) // 3
    
    min_brightness = 255
    water_angle = 0
    
    for angle in range(0, 360, 10):
        rad = np.radians(angle)
        x = int(center_x + radius * np.cos(rad))
        y = int(center_y + radius * np.sin(rad))
        
        # Sample a region
        x1, y1 = max(0, x-25), max(0, y-25)
        x2, y2 = min(w, x+25), min(h, y+25)
        
        if x2 > x1 and y2 > y1:
            region = blurred[y1:y2, x1:x2]
            brightness = np.mean(region)
            if brightness < min_brightness:
                min_brightness = brightness
                water_angle = angle
    
    return water_angle


def calc_canvas_size(w, h, angle_deg):
    """Calculate canvas size needed to fit rotated rectangle."""
    angle_rad = np.radians(abs(angle_deg))
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(h * cos_a + w * sin_a)
    return max(new_w, w), max(new_h, h)


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'samples/sample_short.mp4'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output/stabilized_v3.mp4'
    smooth_window = 8
    
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {input_file} ({orig_w}x{orig_h} @ {fps:.1f} fps, {total} frames)")
    
    # First pass: detect all water angles
    print("Pass 1: Detecting water positions...")
    rotations = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        water_angle = find_water_angle(frame)
        # Rotation needed to put water at bottom (90 degrees in image coords)
        rotation = 90 - water_angle
        # Normalize to -180 to 180
        while rotation > 180: rotation -= 360
        while rotation < -180: rotation += 360
        
        rotations.append(rotation)
    
    cap.release()
    
    # Smooth angles
    smoothed = []
    for i in range(len(rotations)):
        start = max(0, i - smooth_window)
        smoothed.append(np.mean(rotations[start:i+1]))
    
    # Find max rotation to calculate canvas size
    max_abs_rot = max(abs(r) for r in smoothed)
    canvas_w, canvas_h = calc_canvas_size(orig_w, orig_h, max_abs_rot)
    # Add padding
    canvas_w += 50
    canvas_h += 50
    
    print(f"Max rotation: {max_abs_rot:.1f}°")
    print(f"Output canvas: {canvas_w}x{canvas_h}")
    
    # Second pass: apply rotation
    print("Pass 2: Rotating frames...")
    cap = cv2.VideoCapture(input_file)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_avi = output_file.replace('.mp4', '.avi')
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (canvas_w, canvas_h))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        angle = smoothed[frame_idx]
        
        # Create rotation matrix centered on original frame
        center = (orig_w // 2, orig_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Offset to center rotated frame in larger canvas
        offset_x = (canvas_w - orig_w) // 2
        offset_y = (canvas_h - orig_h) // 2
        M[0, 2] += offset_x
        M[1, 2] += offset_y
        
        # Apply rotation to larger canvas
        rotated = cv2.warpAffine(frame, M, (canvas_w, canvas_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        
        out.write(rotated)
        
        if frame_idx % 30 == 0:
            water_dir = 90 - angle
            print(f"  Frame {frame_idx}/{total}  water={water_dir:.0f}°  rotate={angle:.1f}°")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n✅ AVI done: {temp_avi}")
    
    # Convert to MP4
    print(f"Converting to MP4...")
    result = subprocess.run([
        'ffmpeg', '-y', '-i', temp_avi,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        output_file
    ], capture_output=True, text=True)
    
    print(f"✅ Final: {output_file}")


if __name__ == '__main__':
    main()

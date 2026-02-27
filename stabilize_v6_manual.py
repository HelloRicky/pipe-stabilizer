#!/usr/bin/env python3
"""
Pipe Camera Stabilizer v6 - Manual Keyframe Based
Based on Opus visual analysis of the actual frames.

Frame analysis results:
- 0s (frame 0): water at 7-8 o'clock = ~225° in image coords
- 2.5s (frame 73): water at 8 o'clock = ~240°
- 5s (frame 146): water at 10-11 o'clock = ~315°
- 8s (frame 234): water at 5-6 o'clock = ~165°

To put water at bottom (6 o'clock = 90°), we need:
rotation = 90 - water_angle
"""

import cv2
import numpy as np
import subprocess

def main():
    input_file = 'samples/sample_short.mp4'
    output_file = 'output/stabilized_v6.mp4'
    
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {input_file} ({orig_w}x{orig_h} @ {fps:.1f} fps, {total} frames)")
    
    # Keyframes based on Opus visual analysis
    # (frame_number, water_angle_in_degrees)
    # Image coords: 0=right(3oclock), 90=down(6oclock), 180=left(9oclock), 270=up(12oclock)
    keyframes = [
        (0, 225),      # 0s: water at 7-8 o'clock
        (37, 240),     # 1.25s: water moving toward 8 o'clock  
        (73, 240),     # 2.5s: water at 8 o'clock
        (110, 280),    # 3.75s: water moving toward 10 o'clock
        (146, 315),    # 5s: water at 10-11 o'clock
        (180, 270),    # 6s: water moving back
        (210, 200),    # 7s: water heading toward 6 o'clock
        (234, 165),    # 8s: water at 5-6 o'clock
        (278, 150),    # end: water at ~5 o'clock
    ]
    
    # Interpolate water angles for all frames
    water_angles = []
    for frame_idx in range(total):
        # Find surrounding keyframes
        prev_kf = keyframes[0]
        next_kf = keyframes[-1]
        
        for i in range(len(keyframes) - 1):
            if keyframes[i][0] <= frame_idx <= keyframes[i+1][0]:
                prev_kf = keyframes[i]
                next_kf = keyframes[i+1]
                break
        
        # Linear interpolation
        if prev_kf[0] == next_kf[0]:
            angle = prev_kf[1]
        else:
            t = (frame_idx - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            angle = prev_kf[1] + t * (next_kf[1] - prev_kf[1])
        
        water_angles.append(angle)
    
    # Calculate corrections: rotate to put water at bottom (90°)
    corrections = [90 - angle for angle in water_angles]
    
    # Find max correction for canvas size
    max_abs = max(abs(c) for c in corrections)
    rad = np.radians(max_abs)
    canvas_w = int(orig_w * abs(np.cos(rad)) + orig_h * abs(np.sin(rad))) + 50
    canvas_h = int(orig_h * abs(np.cos(rad)) + orig_w * abs(np.sin(rad))) + 50
    
    print(f"Water angle range: {min(water_angles):.0f}° to {max(water_angles):.0f}°")
    print(f"Correction range: {min(corrections):.0f}° to {max(corrections):.0f}°")
    print(f"Canvas size: {canvas_w}x{canvas_h}")
    
    # Process video
    print("\nProcessing frames...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_avi = output_file.replace('.mp4', '.avi')
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (canvas_w, canvas_h))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        correction = corrections[frame_idx]
        
        # Rotation matrix
        center = (orig_w // 2, orig_h // 2)
        M = cv2.getRotationMatrix2D(center, correction, 1.0)
        
        # Offset for larger canvas
        offset_x = (canvas_w - orig_w) // 2
        offset_y = (canvas_h - orig_h) // 2
        M[0, 2] += offset_x
        M[1, 2] += offset_y
        
        rotated = cv2.warpAffine(frame, M, (canvas_w, canvas_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        
        out.write(rotated)
        
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total}  water={water_angles[frame_idx]:.0f}°  rotation={correction:.0f}°")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n✅ AVI done")
    
    # Convert to MP4
    print("Converting to MP4...")
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_avi,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        output_file
    ], capture_output=True)
    
    print(f"✅ Final: {output_file}")


if __name__ == '__main__':
    main()

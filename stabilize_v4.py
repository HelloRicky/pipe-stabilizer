#!/usr/bin/env python3
"""
Pipe Camera Stabilizer v4 - ORB Feature Tracking
Uses OpenCV ORB to track rotation between frames.
"""

import cv2
import numpy as np
import subprocess
import sys

def find_water_angle(frame):
    """Find initial water direction using brightness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    radius = min(h, w) // 3
    
    min_brightness = 255
    water_angle = 90  # default bottom
    
    for angle in range(0, 360, 10):
        rad = np.radians(angle)
        x = int(cx + radius * np.cos(rad))
        y = int(cy + radius * np.sin(rad))
        
        x1, y1 = max(0, x-25), max(0, y-25)
        x2, y2 = min(w, x+25), min(h, y+25)
        
        if x2 > x1 and y2 > y1:
            region = blurred[y1:y2, x1:x2]
            brightness = np.mean(region)
            if brightness < min_brightness:
                min_brightness = brightness
                water_angle = angle
    
    return water_angle


def estimate_rotation_orb(prev_frame, curr_frame, orb, bf):
    """
    Estimate rotation between two frames using ORB feature matching.
    Returns rotation angle in degrees.
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0
    
    # Match features
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Take top matches
    good_matches = matches[:min(50, len(matches))]
    
    if len(good_matches) < 10:
        return 0
    
    # Get matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Estimate rotation using the center of the frame
    h, w = prev_frame.shape[:2]
    cx, cy = w / 2, h / 2
    
    # Calculate angle change for each matched pair
    angles = []
    for p1, p2 in zip(pts1, pts2):
        # Vector from center to point
        v1 = p1 - np.array([cx, cy])
        v2 = p2 - np.array([cx, cy])
        
        # Calculate angle of each vector
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        
        # Angle difference
        diff = np.degrees(a2 - a1)
        # Normalize
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        
        angles.append(diff)
    
    # Use median to be robust to outliers
    if angles:
        return np.median(angles)
    return 0


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'samples/sample_short.mp4'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output/stabilized_v4.mp4'
    
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {input_file} ({orig_w}x{orig_h} @ {fps:.1f} fps, {total} frames)")
    
    # Initialize ORB detector and matcher
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # First pass: track cumulative rotation
    print("Pass 1: Tracking rotation with ORB...")
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video")
        return
    
    # Initial water angle
    initial_water = find_water_angle(first_frame)
    initial_correction = 90 - initial_water  # Rotate to put water at bottom
    print(f"Initial water at {initial_water}°, correction = {initial_correction}°")
    
    cumulative_rotations = [0]  # Camera rotation from first frame
    prev_frame = first_frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Estimate rotation from previous frame
        delta = estimate_rotation_orb(prev_frame, frame, orb, bf)
        
        # Accumulate (camera rotated by delta, so scene appears to rotate opposite)
        cumulative = cumulative_rotations[-1] + delta
        cumulative_rotations.append(cumulative)
        
        prev_frame = frame
    
    cap.release()
    
    print(f"Tracked {len(cumulative_rotations)} frames")
    print(f"Total camera rotation: {cumulative_rotations[-1]:.1f}°")
    
    # Smooth cumulative rotations
    smooth_window = 5
    smoothed = []
    for i in range(len(cumulative_rotations)):
        start = max(0, i - smooth_window)
        smoothed.append(np.mean(cumulative_rotations[start:i+1]))
    
    # Calculate corrections: undo camera rotation + initial water correction
    corrections = [initial_correction - r for r in smoothed]
    
    # Find max rotation for canvas size
    max_abs = max(abs(c) for c in corrections)
    rad = np.radians(max_abs)
    canvas_w = int(orig_w * abs(np.cos(rad)) + orig_h * abs(np.sin(rad))) + 50
    canvas_h = int(orig_h * abs(np.cos(rad)) + orig_w * abs(np.sin(rad))) + 50
    
    print(f"Max correction: {max_abs:.1f}°")
    print(f"Canvas size: {canvas_w}x{canvas_h}")
    
    # Second pass: apply corrections
    print("Pass 2: Applying corrections...")
    cap = cv2.VideoCapture(input_file)
    
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
            print(f"  Frame {frame_idx}/{total}  cam_rot={smoothed[frame_idx]:.1f}°  correction={correction:.1f}°")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n✅ AVI done: {temp_avi}")
    
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

#!/usr/bin/env python3
"""
Pipe Camera Stabilizer v7 - Keyframe Feature Extraction + Tracking
- Extract features every 10 frames (keyframes)
- Track features between frames using optical flow
- Accumulate rotation and correct canvas
"""

import cv2
import numpy as np
import subprocess
import sys

def detect_water_angle(frame):
    """Detect water angle using brightness analysis."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    radius = min(h, w) // 3
    
    min_brightness = 255
    water_angle = 90  # default bottom
    
    for angle in range(0, 360, 5):
        rad = np.radians(angle)
        x = int(cx + radius * np.cos(rad))
        y = int(cy + radius * np.sin(rad))
        
        x1, y1 = max(0, x-20), max(0, y-20)
        x2, y2 = min(w, x+20), min(h, y+20)
        
        if x2 > x1 and y2 > y1:
            region = blurred[y1:y2, x1:x2]
            brightness = np.mean(region)
            if brightness < min_brightness:
                min_brightness = brightness
                water_angle = angle
    
    return water_angle


def track_rotation(prev_gray, curr_gray, prev_pts):
    """
    Track feature points and estimate rotation between frames.
    Returns rotation angle in degrees.
    """
    if prev_pts is None or len(prev_pts) < 10:
        return 0, None
    
    # Lucas-Kanade optical flow
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Filter good points
    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]
    
    if len(good_prev) < 5:
        return 0, None
    
    # Calculate rotation from matched points
    h, w = prev_gray.shape
    cx, cy = w / 2, h / 2
    
    angles = []
    for p1, p2 in zip(good_prev, good_curr):
        # Vector from center to each point
        v1 = p1[0] - np.array([cx, cy])
        v2 = p2[0] - np.array([cx, cy])
        
        # Angle of each vector
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        
        diff = np.degrees(a2 - a1)
        # Normalize to -180 to 180
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        
        angles.append(diff)
    
    # Use median for robustness
    rotation = np.median(angles) if angles else 0
    
    return rotation, good_curr


def extract_features(gray):
    """Extract good features for tracking."""
    # Use Shi-Tomasi corner detection
    features = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    return features


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'samples/sample_short.mp4'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output/stabilized_v7.mp4'
    keyframe_interval = 10
    
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {input_file} ({orig_w}x{orig_h} @ {fps:.1f} fps, {total} frames)")
    print(f"Keyframe interval: every {keyframe_interval} frames")
    
    # Pass 1: Build rotation trajectory
    print("\nPass 1: Building rotation trajectory...")
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video")
        return
    
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect initial water angle
    initial_water = detect_water_angle(first_frame)
    # Correction to put water at bottom (90° in image coords)
    initial_correction = 90 - initial_water
    print(f"Initial water at {initial_water}°, initial correction = {initial_correction}°")
    
    # Initialize tracking
    prev_gray = first_gray
    prev_pts = extract_features(first_gray)
    
    cumulative_camera_rotation = 0
    camera_rotations = [0]  # Camera rotation relative to first frame
    
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track rotation from previous frame
        delta_rotation, tracked_pts = track_rotation(prev_gray, curr_gray, prev_pts)
        
        # Accumulate camera rotation
        cumulative_camera_rotation += delta_rotation
        camera_rotations.append(cumulative_camera_rotation)
        
        # Every keyframe_interval frames, re-extract features
        if frame_idx % keyframe_interval == 0:
            prev_pts = extract_features(curr_gray)
            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total}  camera_rot={cumulative_camera_rotation:.1f}°")
        else:
            # Update tracked points
            prev_pts = tracked_pts if tracked_pts is not None else extract_features(curr_gray)
        
        prev_gray = curr_gray
        frame_idx += 1
    
    cap.release()
    
    print(f"Total camera rotation: {cumulative_camera_rotation:.1f}°")
    
    # Calculate corrections: undo camera rotation + apply initial water correction
    corrections = [initial_correction - rot for rot in camera_rotations]
    
    # Smooth corrections
    smooth_window = 5
    smoothed = []
    for i in range(len(corrections)):
        start = max(0, i - smooth_window)
        smoothed.append(np.mean(corrections[start:i+1]))
    corrections = smoothed
    
    # Find max correction for canvas size
    max_abs = max(abs(c) for c in corrections)
    rad = np.radians(max_abs)
    canvas_w = int(orig_w * abs(np.cos(rad)) + orig_h * abs(np.sin(rad))) + 50
    canvas_h = int(orig_h * abs(np.cos(rad)) + orig_w * abs(np.sin(rad))) + 50
    
    print(f"Correction range: {min(corrections):.0f}° to {max(corrections):.0f}°")
    print(f"Canvas size: {canvas_w}x{canvas_h}")
    
    # Pass 2: Apply corrections
    print("\nPass 2: Applying corrections...")
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
            print(f"  Frame {frame_idx}/{total}  cam_rot={camera_rotations[frame_idx]:.1f}°  correction={correction:.1f}°")
        
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

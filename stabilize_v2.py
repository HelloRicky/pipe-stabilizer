#!/usr/bin/env python3
"""
Pipe Camera Stabilizer v2 - Gravity Detection
Keeps fluid/water at the bottom of the frame by detecting the "down" direction.
"""

import cv2
import numpy as np
import argparse

def find_gravity_angle(frame):
    """
    Detect the gravity direction by finding where the darker region (fluid) is.
    Returns angle in degrees needed to rotate fluid to bottom.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Find the darkest regions (fluid is usually darker)
    # Use adaptive threshold or simple threshold
    _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of dark regions
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    # Find the largest dark region (likely the fluid)
    largest = max(contours, key=cv2.contourArea)
    
    # Get the centroid of the dark region
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return 0
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Calculate angle from center of frame to centroid
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Vector from center to fluid centroid
    dx = cx - center_x
    dy = cy - center_y
    
    # Angle to rotate so fluid is at bottom (270 degrees / -90 degrees)
    current_angle = np.degrees(np.arctan2(dy, dx))
    
    # We want the fluid at the bottom (90 degrees in image coords, where y increases downward)
    target_angle = 90  # Bottom of image
    
    rotation_needed = target_angle - current_angle
    
    # Normalize to -180 to 180
    while rotation_needed > 180:
        rotation_needed -= 360
    while rotation_needed < -180:
        rotation_needed += 360
    
    return rotation_needed


def find_gravity_angle_brightness(frame):
    """
    Alternative: Use brightness gradient to find which side is "down".
    The bottom of a pipe with fluid is usually darker.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    
    h, w = gray.shape
    center_x, center_y = w // 2, h // 2
    
    # Sample brightness in 8 directions
    radius = min(h, w) // 3
    angles = np.linspace(0, 360, 8, endpoint=False)
    
    brightness_by_angle = []
    for angle in angles:
        rad = np.radians(angle)
        x = int(center_x + radius * np.cos(rad))
        y = int(center_y + radius * np.sin(rad))
        
        # Sample a small region
        x1, y1 = max(0, x-20), max(0, y-20)
        x2, y2 = min(w, x+20), min(h, y+20)
        
        if x2 > x1 and y2 > y1:
            region = blurred[y1:y2, x1:x2]
            brightness_by_angle.append((angle, np.mean(region)))
        else:
            brightness_by_angle.append((angle, 128))
    
    # Find the darkest direction (that's where the fluid is)
    darkest = min(brightness_by_angle, key=lambda x: x[1])
    darkest_angle = darkest[0]
    
    # Rotate so darkest is at bottom (90 degrees)
    rotation_needed = 90 - darkest_angle
    
    while rotation_needed > 180:
        rotation_needed -= 360
    while rotation_needed < -180:
        rotation_needed += 360
    
    return rotation_needed


def rotate_frame(frame, angle):
    """Rotate frame with border replication to avoid black corners."""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate new dimensions to avoid cropping
    cos_a = abs(np.cos(np.radians(angle)))
    sin_a = abs(np.sin(np.radians(angle)))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # Adjust rotation matrix for new size
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(frame, M, (new_w, new_h), 
                              borderMode=cv2.BORDER_REPLICATE)
    
    # Crop back to original size from center
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    cropped = rotated[start_y:start_y+h, start_x:start_x+w]
    
    return cropped


def main():
    parser = argparse.ArgumentParser(description='Pipe Camera Stabilizer v2 - Gravity Detection')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', default='output.mp4', help='Output video file')
    parser.add_argument('--smooth', type=int, default=15, help='Smoothing window (frames)')
    parser.add_argument('--method', choices=['dark', 'brightness'], default='brightness',
                        help='Detection method: dark (find dark regions) or brightness (gradient)')
    parser.add_argument('--preview', action='store_true', help='Show preview')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {args.input} ({w}x{h} @ {fps:.1f} fps, {total} frames)")
    print(f"Method: {args.method}")
    print(f"Smooth window: {args.smooth} frames")
    
    # Output to AVI first (more reliable)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_output = args.output.replace('.mp4', '.avi')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))
    
    angles = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect gravity direction
        if args.method == 'dark':
            angle = find_gravity_angle(frame)
        else:
            angle = find_gravity_angle_brightness(frame)
        
        angles.append(angle)
        
        # Smooth the angle
        start = max(0, frame_idx - args.smooth)
        smoothed_angle = np.mean(angles[start:frame_idx+1])
        
        # Rotate frame
        rotated = rotate_frame(frame, smoothed_angle)
        
        # Ensure output size matches
        if rotated.shape[:2] != (h, w):
            rotated = cv2.resize(rotated, (w, h))
        
        out.write(rotated)
        
        if args.preview:
            cv2.imshow('Stabilized', rotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} ({100*frame_idx//total}%)  angle={smoothed_angle:.1f}°")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Done — {frame_idx} frames processed")
    print(f"Temp output: {temp_output}")
    
    # Convert to MP4
    import subprocess
    final_output = args.output
    print(f"Converting to MP4: {final_output}")
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_output,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        final_output
    ], capture_output=True)
    print(f"✅ Final output: {final_output}")


if __name__ == '__main__':
    main()

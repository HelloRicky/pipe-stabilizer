#!/usr/bin/env python3
"""
Pipe Camera Stabilizer v5 - Polar Unwrap + Specular Detection + Kalman Filter
Based on Opus analysis of the pipe inspection video characteristics.
"""

import cv2
import numpy as np
import subprocess
import sys

class KalmanAngleFilter:
    """Kalman filter for smoothing angle estimates."""
    def __init__(self):
        # State: [angle, angular_velocity]
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1]], np.float32) * 0.5
        self.initialized = False
    
    def update(self, angle):
        if not self.initialized:
            self.kf.statePre = np.array([[angle], [0]], np.float32)
            self.kf.statePost = np.array([[angle], [0]], np.float32)
            self.initialized = True
            return angle
        
        self.kf.predict()
        measured = np.array([[np.float32(angle)]])
        self.kf.correct(measured)
        return self.kf.statePost[0, 0]


def find_pipe_center(frame):
    """Find the center of the circular pipe view."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # The center is typically the darkest point (looking down the pipe)
    # Or use Hough circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                                param1=50, param2=30, minRadius=100, maxRadius=300)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Take the largest circle
        largest = max(circles[0], key=lambda c: c[2])
        return (largest[0], largest[1]), largest[2]
    
    # Fallback: use image center
    h, w = frame.shape[:2]
    return (w // 2, h // 2), min(h, w) // 2 - 20


def polar_unwrap(frame, center, radius, num_angles=360, num_radii=100):
    """
    Unwrap the circular pipe view to a rectangular image.
    X-axis = angle (0-360), Y-axis = radius (inner to outer)
    """
    cx, cy = center
    unwrapped = np.zeros((num_radii, num_angles, 3), dtype=np.uint8)
    
    for angle_idx in range(num_angles):
        angle = angle_idx * 2 * np.pi / num_angles
        for r_idx in range(num_radii):
            r = int(radius * 0.3 + (radius * 0.7) * r_idx / num_radii)  # Skip center void
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                unwrapped[r_idx, angle_idx] = frame[y, x]
    
    return unwrapped


def detect_water_angle_specular(unwrapped):
    """
    Detect water position by finding the specular reflection band.
    Returns angle in degrees where the water is.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2HSV)
    
    # Specular reflections: high Value, low-to-medium Saturation
    v_channel = hsv[:, :, 2]
    
    # Find the brightest columns (angles)
    brightness_per_angle = np.mean(v_channel, axis=0)
    
    # Smooth the brightness curve
    kernel_size = 15
    smoothed = np.convolve(brightness_per_angle, np.ones(kernel_size)/kernel_size, mode='same')
    
    # Find the peak (specular reflection = water surface)
    peak_idx = np.argmax(smoothed)
    
    return peak_idx  # This is the angle in degrees (0-359)


def detect_water_angle_darkness(unwrapped):
    """
    Detect water position by finding the darkest region with low texture.
    Water is darker and smoother than dry pipe.
    """
    gray = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean darkness per angle sector
    darkness_per_angle = []
    sector_width = 10
    
    for angle in range(0, 360, sector_width):
        sector = gray[:, angle:angle+sector_width]
        mean_val = np.mean(sector)
        std_val = np.std(sector)
        # Water is dark AND smooth (low std)
        score = mean_val + std_val * 0.5  # Lower is more water-like
        darkness_per_angle.append((angle + sector_width//2, score))
    
    # Find minimum (most water-like)
    min_entry = min(darkness_per_angle, key=lambda x: x[1])
    return min_entry[0]


def detect_water_angle_combined(frame, center, radius):
    """
    Combine multiple detection methods for robust water angle detection.
    """
    unwrapped = polar_unwrap(frame, center, radius)
    
    # Method 1: Specular reflection
    specular_angle = detect_water_angle_specular(unwrapped)
    
    # Method 2: Dark + smooth region
    darkness_angle = detect_water_angle_darkness(unwrapped)
    
    # Method 3: Direct intensity sampling (original approach)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    cx, cy = center
    
    min_brightness = 255
    direct_angle = 0
    sample_radius = int(radius * 0.7)
    
    for angle in range(0, 360, 5):
        rad = np.radians(angle)
        x = int(cx + sample_radius * np.cos(rad))
        y = int(cy + sample_radius * np.sin(rad))
        
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            # Sample a small region
            x1, y1 = max(0, x-15), max(0, y-15)
            x2, y2 = min(frame.shape[1], x+15), min(frame.shape[0], y+15)
            region = blurred[y1:y2, x1:x2]
            brightness = np.mean(region)
            if brightness < min_brightness:
                min_brightness = brightness
                direct_angle = angle
    
    # Weighted combination
    # The specular reflection is often ~90 degrees from the water (at the edge)
    # So we need to adjust - the water is opposite the brightest reflection
    
    # For now, use darkness detection as primary (most reliable for this video)
    # with specular as validation
    
    # Check if specular and darkness are roughly opposite (good sign)
    angle_diff = abs(specular_angle - darkness_angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    if angle_diff > 90:  # They're roughly opposite - use darkness
        return darkness_angle
    else:
        # Average them with weights
        return int((darkness_angle * 0.6 + direct_angle * 0.4) % 360)


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'samples/sample_short.mp4'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output/stabilized_v5.mp4'
    
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {input_file} ({orig_w}x{orig_h} @ {fps:.1f} fps, {total} frames)")
    print("Using: Polar unwrap + Specular/Darkness detection + Kalman filter")
    
    # First pass: detect water angles
    print("\nPass 1: Detecting water positions...")
    kalman = KalmanAngleFilter()
    water_angles = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find pipe center and radius
        center, radius = find_pipe_center(frame)
        
        # Detect water angle
        raw_angle = detect_water_angle_combined(frame, center, radius)
        
        # Apply Kalman filter for smoothing
        # Handle angle wraparound
        if water_angles:
            prev = water_angles[-1]
            # Unwrap angle to avoid jumps across 0/360 boundary
            while raw_angle - prev > 180:
                raw_angle -= 360
            while raw_angle - prev < -180:
                raw_angle += 360
        
        filtered_angle = kalman.update(raw_angle)
        water_angles.append(filtered_angle)
        
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total}  raw={raw_angle:.0f}°  filtered={filtered_angle:.0f}°")
        
        frame_idx += 1
    
    cap.release()
    
    # Calculate corrections (rotate water to 270° = bottom in image coords)
    # In image coordinates: 0°=right, 90°=down, 180°=left, 270°=up
    # We want water at 90° (bottom)
    corrections = [90 - angle for angle in water_angles]
    
    # Find max correction for canvas sizing
    max_abs = max(abs(c) for c in corrections)
    rad = np.radians(max_abs)
    canvas_w = int(orig_w * abs(np.cos(rad)) + orig_h * abs(np.sin(rad))) + 50
    canvas_h = int(orig_h * abs(np.cos(rad)) + orig_w * abs(np.sin(rad))) + 50
    
    print(f"\nWater angle range: {min(water_angles):.0f}° to {max(water_angles):.0f}°")
    print(f"Max correction: {max_abs:.0f}°")
    print(f"Canvas size: {canvas_w}x{canvas_h}")
    
    # Second pass: apply corrections
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
            print(f"  Frame {frame_idx}/{total}  water={water_angles[frame_idx]:.0f}°  correction={correction:.0f}°")
        
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

#!/usr/bin/env python3
"""
Pipe Camera Video Stabilizer
Detects dominant horizontal lines (pipe edges, water level) and corrects rotation.
"""

import argparse
import sys
from collections import deque

import cv2
import numpy as np


def detect_rotation_angle(frame):
    """
    Detect the dominant rotation angle in a frame using Hough Line Transform.
    Returns the angle (degrees) needed to level the horizon, or None if undetermined.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=frame.shape[1] // 6,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return None

    angles = []
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        angle = np.degrees(np.arctan2(dy, dx))

        # Normalise to -90..+90
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # Only consider lines reasonably horizontal (within ±45°)
        if abs(angle) <= 45:
            angles.append(angle)
            lengths.append(length)

    if not angles:
        return None

    # Weighted median by line length — robust against outliers
    order = np.argsort(angles)
    sorted_angles = np.array(angles)[order]
    sorted_weights = np.array(lengths)[order]
    cumw = np.cumsum(sorted_weights)
    median_angle = sorted_angles[np.searchsorted(cumw, cumw[-1] / 2)]

    return float(median_angle)


def rotate_frame(frame, angle):
    """Rotate frame by angle degrees, expanding canvas to avoid cropping corners."""
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(frame, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def smooth_angle(buffer, new_angle):
    """Add new_angle to rolling buffer, return smoothed value."""
    if new_angle is not None:
        buffer.append(new_angle)
    if not buffer:
        return 0.0
    return float(np.mean(buffer))


def stabilize(input_path, output_path, smooth_window=30, preview=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input : {input_path}  ({orig_w}x{orig_h} @ {fps:.1f} fps, {total} frames)")
    print(f"Output: {output_path}")
    print(f"Smooth window: {smooth_window} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    angle_buffer = deque(maxlen=smooth_window)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_angle = detect_rotation_angle(frame)
            correction = -smooth_angle(angle_buffer, raw_angle)
            stabilized = rotate_frame(frame, correction)

            if writer is None:
                out_h, out_w = stabilized.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
                print(f"Output size: {out_w}x{out_h}")

            writer.write(stabilized)

            if preview:
                disp = stabilized
                out_w_d = stabilized.shape[1]
                if out_w_d > 1280:
                    scale = 1280 / out_w_d
                    disp = cv2.resize(stabilized, None, fx=scale, fy=scale)
                cv2.imshow("Stabilized Preview  (q = quit)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Preview closed.")
                    break

            frame_idx += 1
            if frame_idx % 50 == 0 or frame_idx == total:
                pct = (frame_idx / total * 100) if total > 0 else 0
                ad = f"{raw_angle:.1f}" if raw_angle is not None else "N/A"
                print(f"  Frame {frame_idx}/{total} ({pct:.0f}%)  detected={ad}°  correction={correction:.1f}°")

    finally:
        cap.release()
        if writer:
            writer.release()
        if preview:
            cv2.destroyAllWindows()

    print(f"\n✅  Done — {frame_idx} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stabilize pipe camera footage by correcting rotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input video (MP4, AVI, …)")
    parser.add_argument("-o", "--output", default="stabilized.mp4", help="Output video file")
    parser.add_argument("--smooth", type=int, default=30, metavar="N",
                        help="Smoothing window (frames to average rotation over)")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview (requires display)")
    args = parser.parse_args()
    stabilize(args.input, args.output, smooth_window=args.smooth, preview=args.preview)


if __name__ == "__main__":
    main()

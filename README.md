# 🔩 Pipe Camera Video Stabilizer

POC tool that auto-corrects rotation in pipe camera footage by detecting the dominant horizontal line (pipe edges, water level) and applying a counter-rotation per frame.

## How it works

1. Each frame is converted to greyscale and edge-detected (Canny)
2. Hough Line Transform finds all prominent lines
3. Lines within ±45° of horizontal are kept; a length-weighted median gives the dominant tilt angle
4. A rolling window smooths the angle over time (avoids jitter)
5. Each frame is counter-rotated and written to the output video

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python stabilize.py input.mp4 -o output.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | `stabilized.mp4` | Output video path |
| `--smooth N` | `30` | Smoothing window (frames). Higher = smoother but slower to adapt |
| `--preview` | off | Show live preview window while processing |

### Examples

```bash
# Basic stabilization
python stabilize.py pipe_footage.mp4 -o pipe_stable.mp4

# Aggressive smoothing (stable camera movement)
python stabilize.py pipe_footage.mp4 -o pipe_stable.mp4 --smooth 60

# Fast-moving camera, less smoothing
python stabilize.py pipe_footage.mp4 -o pipe_stable.mp4 --smooth 10

# Watch it process in real-time
python stabilize.py pipe_footage.mp4 --preview
```

## Notes

- Output canvas is slightly larger than input to avoid black corners after rotation
- `BORDER_REPLICATE` fills edges — keeps the pipe wall texture visible rather than black bars
- Works best when the pipe has a clear horizontal feature (water level, pipe seam, or strong edge)
- If a frame has no detectable line, the last known angle is reused from the smoothing buffer

## Roadmap (post-POC)

- [ ] GPU acceleration (cv2.cuda)
- [ ] Audio passthrough
- [ ] Crop-to-original-size option
- [ ] Confidence threshold tuning
- [ ] Support for circular pipe detection (Hough Circles)

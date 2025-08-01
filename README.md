# Face Detection & Recognition Tool

Real-time face detection and recognition system using Mac camera with automatic script triggering.

## Features

- 🎥 Real-time Mac camera face detection
- 🔍 High-accuracy face recognition and comparison
- 📊 Confidence scoring with distance metrics
- 🖼️ Multiple image formats (JPG, JPEG, PNG, BMP)
- ⚡ Real-time processing and display
- 🔄 Hot-reload reference images
- 🚀 **Smart Script Triggering**: Auto-execute scripts on face match

## Requirements

- macOS
- Python 3.8+
- Camera access
- UV package manager

## Installation

### 1. Install UV Package Manager

```bash
# Using Homebrew
brew install uv

# Or using pip
pip install uv
```

### 2. Setup Project

```bash
cd your-workspace
git clone <repository-url>
cd detect-hui-face
```

### 3. Install Dependencies

```bash
# Create virtual environment and install dependencies
uv sync
```

### 4. Add Reference Images

Add face images to the `face/` directory:

```bash
face/
├── person1.jpg
├── person2.png
└── person3.jpeg
```

**Notes**:
- Filename becomes the person's label (without extension)
- Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Use clear, well-lit, front-facing photos
- One main face per image

## Usage

### Run the Program

```bash
uv run python face_detector.py
```

### How it Works

1. **Camera Launch**: Automatically accesses system camera
2. **Real-time Detection**: Shows green boxes (known faces) or red boxes (unknown faces)
3. **Results**: 
   - Green box + name: Successful match
   - Red box + "Unknown": No match found
   - Distance number: Similarity score (lower = more similar)

### Controls

- `q`: Quit
- `r`: Reload reference images
- `+`: Increase tolerance (less strict)
- `-`: Decrease tolerance (more strict)

### Sample Output

```
==================================================
Face Detection and Comparison System
==================================================
Loading 3 reference images...
✓ Loaded: john, mary, david
Successfully loaded 3 reference faces

🎯 Recognition tolerance: 0.44 (lower = more strict)
📊 Confidence: <0.3=VERY HIGH, <0.4=HIGH, <0.5=MEDIUM, <0.6=LOW
🚀 Trigger script: ~/scripts/raycast/hui.sh
⏱️  Cooldown: 5 seconds

Controls: 'q'=quit, 'r'=reload, '+'=less strict, '-'=more strict
==================================================

Face detected: john (distance: 0.352) - MATCH - Confidence: HIGH
🚀 Executing trigger script for: john
✅ Script executed successfully

Face detected: stranger (distance: 0.523) - NO MATCH - Confidence: MEDIUM
⚠️  Rejected: distance 0.523 >= threshold 0.44
```

## Configuration

Adjust parameters in `face_detector.py`:

```python
detector = FaceDetector(
    face_dir="face",                          # Reference images directory
    tolerance=0.44,                           # Recognition threshold (0.0-1.0, lower=stricter)
    trigger_script="~/scripts/raycast/hui.sh", # Script to execute on match
    cooldown_seconds=5                        # Minimum seconds between executions
)
```

### Script Triggering

When a face matches, the system automatically executes the specified script:

- **Script path**: Default `~/scripts/raycast/hui.sh`
- **Parameter**: Matched person's name passed as first argument
- **Cooldown**: Prevents rapid re-execution (5 seconds)
- **Timeout**: 30-second execution limit
- **Error handling**: Script failures don't interrupt face detection

#### Example Script

Create `~/scripts/raycast/hui.sh`:

```bash
#!/bin/bash
MATCHED_NAME="$1"

# Send macOS notification
osascript -e "display notification \"User detected: $MATCHED_NAME\" with title \"Face Recognition\""

# Log to file
echo "$(date): User $MATCHED_NAME verified" >> ~/logs/face_detection.log

exit 0
```

Make executable:
```bash
chmod +x ~/scripts/raycast/hui.sh
```

### Threshold Tuning

Recognition accuracy depends on the `tolerance` threshold:

#### Recommended Values:
- `0.3`: Very strict - almost identical faces only
- `0.4-0.44`: Strict - good balance (default)
- `0.5`: Balanced - moderate accuracy
- `0.6+`: Loose - higher false positive risk

#### Real-time Adjustment:
1. Watch distance values in console output
2. Use keyboard shortcuts:
   - `+`: Increase tolerance (if your face gets rejected)
   - `-`: Decrease tolerance (if strangers get accepted)

#### Distance Interpretation:
- **< 0.3**: VERY HIGH confidence (perfect match)
- **0.3-0.4**: HIGH confidence (reliable match)
- **0.4-0.5**: MEDIUM confidence (caution needed)
- **0.5-0.6**: LOW confidence (likely false positive)
- **> 0.6**: VERY LOW confidence (should reject)

#### Adjustment Guide:

**If strangers are accepted:**
```python
detector = FaceDetector(tolerance=0.35)  # More strict
```

**If you get rejected:**
```python  
detector = FaceDetector(tolerance=0.48)  # Less strict
```

#### Best Practice:
1. Start with default (0.44)
2. Test with your face, observe distance values
3. Adjust threshold based on your typical distance
4. Test multiple times for consistency

## Reference Images Guide

### Multiple vs Single Images

**Recommended: 2-3 images per person** for:
1. Better accuracy across different angles/expressions
2. Improved robustness under varying lighting
3. Reduced false positives with diverse training data

### File Organization

**Multiple images per person:**
```
face/
├── john_front.jpg    # Front-facing
├── john_side.jpg     # Side profile  
└── john_smile.jpg    # Different expression
```

**Single image per person:**
```
face/
├── john.jpg
├── mary.jpg
└── david.jpg
```

### Image Quality Requirements

- ✅ **Clarity**: High resolution, sharp facial features
- ✅ **Lighting**: Even, well-lit (avoid harsh shadows)
- ✅ **Angle**: Front-facing or slight profile
- ✅ **Size**: Face occupies significant portion of image
- ❌ **Avoid**: Blurry, dark, obscured, or group photos

## Troubleshooting

### Common Issues

**Camera Access Problems:**
```
Error: Could not open camera!
```
- Check if camera is used by other apps
- Verify camera permissions in System Preferences

**Continuity Camera Issues:**
```
AVCaptureDeviceTypeExternal is deprecated
```
- Program may access iPhone camera instead of Mac built-in
- Solution: Disconnect iPhone or disable Continuity Camera

**No Reference Images:**
```
Error: No reference images found in 'face' directory!
```
- Ensure `face/` directory exists
- Check supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Verify images contain clear faces

**Poor Recognition:**
- Adjust `tolerance` parameter
- Add more reference images with different angles
- Improve camera lighting

### Installation Issues

If dependencies fail to install:

```bash
# Force reinstall problematic packages
uv pip install --force-reinstall opencv-python face-recognition

# Install system dependencies if needed
brew install cmake
```

## Development

For development and debugging:

```bash
# Install development dependencies
uv sync --dev

# Run tests (if available)
uv run pytest

# Format code
uv run black face_detector.py
```

## Technical Stack

- **OpenCV**: Camera handling and image processing
- **face_recognition**: Face detection and feature extraction
- **NumPy**: Numerical computations
- **Pillow**: Image file processing

## License

This project is for educational and personal use only. Please comply with applicable laws and respect privacy rights.

---

**Important Notes**:
- macOS will request camera permissions on first run
- The program displays live video - be mindful of privacy
- Face recognition is for authentication purposes only
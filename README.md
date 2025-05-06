# YOLO v11 Python Annotation Tool

A powerful Python-based annotation tool for creating YOLO format datasets, built with PySide6.

## Features

- **Interactive Annotation**: Draw bounding boxes with mouse, resize and adjust positions
- **Multi-Model Support**: Load and use multiple YOLO models simultaneously
- **Class Selection**: Color-coded bounding boxes with customizable class selection
- **Organization Tools**:
  - Sort images by annotation status (empty first, most/least annotations)
  - Jump to specific images
  - Navigation controls for continuous scrolling
- **Dataset Management**:
  - Add photos without replacing existing ones
  - Remove and save annotated images
  - Detect and remove duplicate images using perceptual hashing
- **Performance Optimizations**:
  - Parallel model loading
  - Batch processing for faster inference
  - Image caching for smoother navigation
  - Status tracking for monitoring model performance

## Quick Start

### Using the GUI Launcher (Recommended)

1. Run the launcher script:
   ```
   python launcher.py
   ```

2. The launcher will:
   - Check your Python installation
   - Verify the virtual environment
   - Install dependencies if needed
   - Provide a one-click "Run" button to start the tool

### Using the Runner Scripts

For macOS/Linux:
```
./run.sh
```

For Windows:
```
run.bat
```

These scripts will automatically set up the virtual environment if needed before running the tool.

## Manual Installation

1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the annotation tool:
   ```
   python Annotator/Yolo\ V11\ Python\ Annotator.py
   ```

## Usage

### Basic Workflow

1. Click "Open Images" to load a set of images
2. Click "Load Model" to select a YOLO model file (.pt)
3. Use "Auto-Annotate Current" or "Auto-Annotate All" to detect objects
4. Manually add, adjust, or delete bounding boxes as needed
5. Change box classes using the class selector
6. Save annotations with "Save Annotations"

### Working with Multiple Models

Use "Try All Models" to:
- Select a folder containing multiple YOLO model files
- Choose which classes to detect
- The tool will try each model in sequence to annotate your images

### Keyboard Shortcuts

- **←/→**: Navigate between images
- **Ctrl+S**: Save annotations
- **Ctrl+C**: Clear annotations on current image
- **Ctrl+A**: Auto-annotate current image
- **Ctrl+D**: Delete current image
- **0-9**: Quick-select box class

## Troubleshooting

- **Memory Issues**: When working with high-resolution images or large models, reduce the batch size in the code
- **Missing Modules**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **CUDA Errors**: For GPU acceleration, ensure compatible PyTorch, CUDA, and GPU drivers

## Requirements

- Python 3.6+
- PySide6
- Ultralytics YOLO
- NumPy
- ImageHash (for duplicate detection)
- PIL (Pillow) 
#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Make script file executable
echo "Making script executable..."
chmod +x Annotator/Yolo\ V11\ Python\ Annotator.py

echo "Installation complete! To run the tool:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the tool: python3 Annotator/Yolo\ V11\ Python\ Annotator.py" 
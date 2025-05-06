#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running install script first..."
    ./install.sh
fi

# Activate virtual environment
source venv/bin/activate

# Run the tool
python3 Annotator/Yolo\ V11\ Python\ Annotator.py

# Deactivate virtual environment when done
deactivate 
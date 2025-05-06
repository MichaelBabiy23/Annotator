@echo off

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Running install script first...
    call install.bat
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the tool
python Annotator\Yolo V11 Python Annotator.py

REM Deactivate virtual environment when done
call deactivate 
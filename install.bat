@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Installation complete! To run the tool:
echo   1. Activate the virtual environment: venv\Scripts\activate
echo   2. Run the tool: python Annotator\Yolo V11 Python Annotator.py
echo.
pause 
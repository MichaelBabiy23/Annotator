#!/usr/bin/env python3
"""
Simple runner script for YOLO Annotation Tool
"""
import os
import sys
import subprocess
import platform

def main():
    """Main entry point"""
    print("YOLO Annotation Tool Runner")
    print("==========================")
    
    # Check if virtual environment exists
    if not os.path.isdir("venv"):
        print("\nVirtual environment not found. Setting up now...")
        
        # Run installation script based on platform
        if platform.system() == "Windows":
            subprocess.run(["install.bat"], shell=True)
        else:
            subprocess.run(["./install.sh"], shell=True)
            
        # Check if installation was successful
        if not os.path.isdir("venv"):
            print("Failed to create virtual environment. Please run install script manually.")
            return 1
    
    # Determine Python command
    if platform.system() == "Windows":
        python_cmd = os.path.join("venv", "Scripts", "python")
        script_path = os.path.join("Annotator", "Yolo V11 Python Annotator.py")
    else:
        python_cmd = os.path.join("venv", "bin", "python3")
        script_path = os.path.join("Annotator", "Yolo V11 Python Annotator.py")
    
    print(f"\nLaunching YOLO Annotation Tool...")
    
    # Run the application
    try:
        result = subprocess.run([python_cmd, script_path])
        return result.returncode
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
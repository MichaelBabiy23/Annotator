from setuptools import setup, find_packages

setup(
    name="yolo-annotator",
    version="1.0.0",
    description="YOLO v11 Annotation Tool for creating YOLO format datasets",
    author="Michael Babiy",
    packages=find_packages(),
    install_requires=[
        "PySide6>=6.5.0",
        "ultralytics>=8.0.0",
        "numpy==1.26.4",
        "imagehash>=4.3.1",
        "Pillow>=9.0.0",
        "PyWavelets>=1.4.1",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "scipy>=1.10.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 
from setuptools import setup, find_packages

setup(
    name="reefscape_yolov12",
    version="0.1.0",
    packages=find_packages(include=["reefscape_yolo", "reefscape_yolo.*"]),
    install_requires=[
        "ultralytics>=8.3.2",
        "matplotlib>=3.7.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.4.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "scipy>=1.10.0",
        "flask>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "reefscape-train=reefscape_yolo.cli.train:main",
            "reefscape-predict=reefscape_yolo.cli.predict:main",
            "reefscape-validate=reefscape_yolo.cli.validate:main",
            "reefscape-api=reefscape_yolo.cli.api:main",
        ],
    },
    python_requires=">=3.8",
    author="Team1100",
    description="REEFSCAPE detection and classification using YOLOv12",
    keywords="yolo, object detection, reefscape",
) 
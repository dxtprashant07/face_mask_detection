# Face Mask Detection System

An end-to-end face mask detection system using YOLOv8 for real-time surveillance and image analysis. This project detects whether individuals are wearing masks correctly, incorrectly, or not at all, making it suitable for public health monitoring and academic research.

## Features

- **Data Preparation**: Convert Pascal VOC datasets to YOLO format
- **Model Training**: Train custom YOLOv8 models on face mask datasets
- **Model Evaluation**: Comprehensive evaluation metrics and visualization
- **Model Export**: Export trained models to ONNX and other formats for deployment
- **Real-time Detection**: OpenCV-based video and image detection
- **Web Interface**: Streamlit app for easy image upload and live webcam detection
- **Pre-trained Models**: Includes trained models for immediate use

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/dxtprashant07/face_mask_detection.git
   cd face_mask_detection
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv yolov8_env
   yolov8_env\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
```bash
python model/train.py
```

### Evaluating the Model
```bash
python model/evaluate.py
```

### Running Inference on Images
```bash
python inference/predict_image.py --image path/to/image.jpg
```

### Running Inference on Videos
```bash
python inference/predict_video.py --video path/to/video.mp4
```

### Launching the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
face_mask_detection/
├── app/
│   └── streamlit_app.py          # Web interface
├── inference/
│   ├── predict_image.py          # Image prediction
│   ├── predict_video.py          # Video prediction
│   └── visualize.py              # Visualization utilities
├── model/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── export.py                 # Model export
├── utils/
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocess.py             # Preprocessing functions
│   ├── metrics.py                # Evaluation metrics
│   └── voc_to_yolo.py            # VOC to YOLO conversion
├── best_face_mask.pt             # Trained PyTorch model
├── best_face_mask.onnx           # Exported ONNX model
├── requirements.txt              # Python dependencies
├── main.py                       # Main entry point
├── test.py                       # Testing script
└── README.md                     # This file
```

## Model Details

- **Architecture**: YOLOv8
- **Classes**: 
  - With Mask
  - Without Mask
  - Mask Worn Incorrectly
- **Input Size**: 640x640
- **Confidence Threshold**: Adjustable (default: 0.6)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit for the web interface
- OpenCV for computer vision tasks

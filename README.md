# Substation Elements Detection

This repository contains code and experiments for **automatic detection of substation elements** (Busbars, Circuit Breakers, Transformers, etc.) using [YOLOv8](https://github.com/ultralytics/ultralytics) and a Gradio-based demo app.

## 🛠 Features
- Train a YOLOv8 model on a custom dataset of substation components.
- Evaluate model performance with validation/test sets.
- Export trained model weights (`best.pt`).
- Interactive **Gradio UI**:
  - Upload an image.
  - Run inference to visualize detected substation components.
  - View/save annotated output.
    
## 📋 Requirements
- Python 3.8+
- [Ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8)
- PyTorch (CUDA if available)
- OpenCV
- Gradio
- PIL, NumPy

Install with:
```bash
pip install ultralytics torch torchvision torchaudio
pip install opencv-python gradio pillow

🚀 Usage
## 1. Clone the repository

git clone https://github.com/<your-username>/substation_elements_detection.git
cd substation_elements_detection

## 2. Run the notebook

Open in Jupyter/Colab and execute step by step:
jupyter notebook substation_elements_detection.ipynb


## 3. Launch the Gradio demo

If you already have trained weights (best.pt), you can run the app to test images interactively:

from ultralytics import YOLO
import gradio as gr

model = YOLO("runs/yolov8_train/weights/best.pt")

def infer(img):
    results = model(img)
    return results[0].plot()[:, :, ::-1]  # returns RGB numpy array

gr.Interface(fn=infer, inputs="image", outputs="image").launch()

import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
from pathlib import Path

# ----------------------
# Path Configuration
# ----------------------
BASE_DIR = Path("D:/rajesh/python/substation_elements_detection")
MODEL_PATH = BASE_DIR / "runs" / "yolov8_train" / "weights" / "best.pt"

# ----------------------
# Load Model
# ----------------------
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(MODEL_PATH))
print(f"[INFO] Loaded model: {MODEL_PATH} on device {device}")

# ----------------------
# Detection Function
# ----------------------
def detect(image: np.ndarray, conf: float = 0.25, imgsz: int = 640):
    results = model.predict(
        source=image,
        imgsz=int(imgsz),
        conf=float(conf),
        device=device,
        verbose=False
    )
    annotated = results[0].plot()          # BGR np.ndarray
    annotated_rgb = annotated[..., ::-1]   # Convert to RGB
    pil_img = Image.fromarray(annotated_rgb)
    return pil_img

# ----------------------
# Gradio UI
# ----------------------
demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence Threshold"),
        gr.Slider(320, 1280, value=640, step=32, label="Image Size")
    ],
    outputs=gr.Image(type="pil", label="Annotated Output"),
    title="Substation Elements Detection Tool",
    description="Upload a substation image. The model will annotate components like CBDS, CT, FSW, etc.",
)

# ----------------------
# Run App
# ----------------------
if __name__ == "__main__":
    demo.queue().launch(debug=True, share=False)


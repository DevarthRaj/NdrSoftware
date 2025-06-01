from roboflow import Roboflow
from ultralytics import YOLO

# Step 1: Download dataset from Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("devarthraj").project("plant-stress-detection-biwkz")
version = project.version(1)
dataset = version.download("yolov8")

# Step 2: Train YOLOv8 using the local dataset path
model = YOLO("yolov8n.pt")  # use yolov8s.pt for better accuracy

model.train(
    data="Plant-stress-detection-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)




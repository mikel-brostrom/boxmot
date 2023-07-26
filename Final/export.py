from ultralytics import YOLO

# Load a model
model = YOLO('./ckpt_saved_model/yolov8/yolov8s.pt')  # load a custom trained

# Export the model
model.export(format='engine', device=0, half=True, simplify=True, imgsz=640)
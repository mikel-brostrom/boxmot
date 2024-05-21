from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="obj.yaml", epochs=100, imgsz=1920,batch=-1)


if __name__ == '__main__':
    main()

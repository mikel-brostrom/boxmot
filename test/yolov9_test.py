import yolov9
import cv2
import torchvision.transforms as transforms

# load pretrained or custom model
model = yolov9.load(
    "/home/mikel.brostrom/Downloads/yolov9-c.pt",
    device="cpu",
)

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = None  # (optional list) filter by class

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor


img = cv2.imread("/home/mikel.brostrom/yolo_tracking/env/lib/python3.10/site-packages/ultralytics/assets/bus.jpg", cv2.IMREAD_COLOR)
tensor = transform(img).unsqueeze(0)

# perform inference
results = model(tensor)

# inference with larger input size and test time augmentation
#results = model(img, size=640)

# parse results
print(len(results.pred))
predictions = results.pred[0]
print(predictions.shape)

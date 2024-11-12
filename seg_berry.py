from ultralytics import YOLO

model = YOLO(r'/home/xplv/huanghanyang/boxmot/tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')
result = model.predict(source=r'/home/xplv/huanghanyang/Track_Datasets/test/aiwei_2_cut.mp4',
                       iou=0.7, conf=0.5, save=True, agnostic_nms=True)


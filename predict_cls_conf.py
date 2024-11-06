import numpy as np

from ultralytics import YOLO
import glob

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO('tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')

    # 设置图像路径
    # path = r'D:\ultralytics\save\CVPPA'
    # path = r'D:\ultralytics\save\aiwei'
    # path = r'D:\ultralytics\save\aiwei_1'
    # path = r'D:\ultralytics\save\Sick'
    # path = r'D:\ultralytics\save\healthy'
    path = r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/save_1.mp4'
    # path = r'save'
    results = model.predict(source=path, save=True, show=False, agnostic_nms=True, show_labels=True,
                            show_conf=True, show_boxes=True, conf=0.1)

    # path = r'D:\华毅\叶片数据集制作\camera\aiwei_9_14\realsense_image_20240914_152528.png'
    files = glob.iglob(path)
    sorted_files=sorted(files)
    count = 0
    # for file in sorted_files:
    #     # 开始检测并保存结果
    #     # results = model.predict(source=file, save=True, show=False, conf=0.1)
    #     results = model.predict(source=file, save=True, show=False, agnostic_nms=True, show_labels=False,
    #                             show_conf=False, show_boxes=False)
        # res = results[0]
        # cls = res.boxes.cls.cpu().numpy().astype(np.int8).tolist()
        # if 6 in cls:
        #     file_name = file.split("/")[-1]
        #     print(file_name)
        #     f.write(file_name)
        #     f.write("\n")



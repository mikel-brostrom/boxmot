# coding=utf-8
# Description:  visualize yolo label image.

import argparse
import os
import cv2
import numpy as np

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]


def main(args):
    img_dir, label_dir, class_names = args.img_dir, args.label_dir, args.class_names

    label_map = dict()
    for class_id, classname in enumerate(class_names):
        label_map[class_id] = classname

    for file in os.listdir(img_dir):
        if file.split('.')[-1] not in IMG_FORMATS:
            print(f'[Warning]: Non-image file {file}')
            continue
        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(label_dir, file[: file.rindex('.')] + '.txt')

        try:
            img_data = cv2.imread(img_path)
            height, width, _ = img_data.shape
            color = [tuple(np.random.choice(range(256), size=3)) for i in class_names]
            thickness = 2

            with open(label_path, 'r') as f:
                for bbox in f:
                    cls, x_c, y_c, w, h = [float(v) if i > 0 else int(v) for i, v in enumerate(bbox.split('\n')[0].split(' '))]

                    x_tl = int((x_c - w / 2) * width)
                    y_tl = int((y_c - h / 2) * height)
                    cv2.rectangle(img_data, (x_tl, y_tl), (x_tl + int(w * width), y_tl + int(h * height)), tuple([int(x) for x in color[cls]]), thickness)
                    cv2.putText(img_data, label_map[cls], (x_tl, y_tl - 10), cv2.FONT_HERSHEY_COMPLEX, 1, tuple([int(x) for x in color[cls]]), thickness)

            cv2.imshow('image', img_data)
            cv2.waitKey(0)
        except Exception as e:
            print(f'[Error]: {e} {img_path}')
    print('======All Done!======')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='VOCdevkit/voc_07_12/images')
    parser.add_argument('--label_dir', default='VOCdevkit/voc_07_12/labels')
    parser.add_argument('--class_names', default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

    args = parser.parse_args()
    print(args)

    main(args)

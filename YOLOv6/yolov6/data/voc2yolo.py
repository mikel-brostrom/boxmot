import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import shutil
import argparse

# VOC dataset (refer https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml)
# VOC2007 trainval: 446MB, 5012 images
# VOC2007 test:     438MB, 4953 images
# VOC2012 trainval: 1.95GB, 17126 images

VOC_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
             'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh
    in_file = open(os.path.join(path, f'VOC{year}/Annotations/{image_id}.xml'))
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in VOC_NAMES and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = VOC_NAMES.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


def gen_voc07_12(voc_path):
    '''
    Generate voc07+12 setting dataset:
    train: # train images 16551 images
        - images/train2012
        - images/train2007
        - images/val2012
        - images/val2007
    val: # val images (relative to 'path')  4952 images
        - images/test2007
    '''
    dataset_root = os.path.join(voc_path, 'voc_07_12')
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    dataset_settings = {'train': ['train2007', 'val2007', 'train2012', 'val2012'], 'val':['test2007']}
    for item in ['images', 'labels']:
        for data_type, data_list in dataset_settings.items():
            for data_name in data_list:
                ori_path = os.path.join(voc_path, item, data_name)
                new_path = os.path.join(dataset_root, item, data_type)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                print(f'[INFO]: Copying {ori_path} to {new_path}')
                for file in os.listdir(ori_path):
                    shutil.copy(os.path.join(ori_path, file), new_path)


def main(args):
    voc_path = args.voc_path
    for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
        imgs_path = os.path.join(voc_path, 'images', f'{image_set}')
        lbs_path = os.path.join(voc_path, 'labels', f'{image_set}')

        try:
            with open(os.path.join(voc_path, f'VOC{year}/ImageSets/Main/{image_set}.txt'), 'r') as f:
                image_ids = f.read().strip().split()
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)
            if not os.path.exists(lbs_path):
                os.makedirs(lbs_path)

            for id in tqdm(image_ids, desc=f'{image_set}{year}'):
                f = os.path.join(voc_path, f'VOC{year}/JPEGImages/{id}.jpg')  # old img path
                lb_path = os.path.join(lbs_path, f'{id}.txt')  # new label path
                convert_label(voc_path, lb_path, year, id)  # convert labels to YOLO format
                if os.path.exists(f):
                    shutil.move(f, imgs_path)       # move image
        except Exception as e:
            print(f'[Warning]: {e} {year}{image_set} convert fail!')

    gen_voc07_12(voc_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_path', default='VOCdevkit')

    args = parser.parse_args()
    print(args)

    main(args)

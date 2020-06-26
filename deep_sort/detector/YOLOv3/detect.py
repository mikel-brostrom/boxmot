import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from yolo_utils import *
from darknet import Darknet

import cv2

namesfile=None
def detect(cfgfile, weightfile, imgfolder):
    m = Darknet(cfgfile)

    #m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    # if m.num_classes == 20:
    #     namesfile = 'data/voc.names'
    # elif m.num_classes == 80:
    #     namesfile = 'data/coco.names'
    # else:
    #     namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    imgfiles = [x for x in os.listdir(imgfolder) if x[-4:] == '.jpg']
    imgfiles.sort()
    for imgname in imgfiles:
        imgfile = os.path.join(imgfolder,imgname)
        
        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((m.width, m.height))

        #for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
            #if i == 1:
        print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(namesfile)
        img = plot_boxes(img, boxes, 'result/{}'.format(os.path.basename(imgfile)), class_names)
        img = np.array(img)
        cv2.imshow('{}'.format(os.path.basename(imgfolder)), img)
        cv2.resizeWindow('{}'.format(os.path.basename(imgfolder)), 1000,800)
        cv2.waitKey(1000)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfolder = sys.argv[3]
        cv2.namedWindow('{}'.format(os.path.basename(imgfolder)), cv2.WINDOW_NORMAL )
        cv2.resizeWindow('{}'.format(os.path.basename(imgfolder)), 1000,800)
        globals()["namesfile"] = sys.argv[4]
        detect(cfgfile, weightfile, imgfolder)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfolder names')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)

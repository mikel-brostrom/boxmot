""""""
import sys
from os.path import dirname, abspath, isfile

sys.path.append(dirname(dirname(abspath(__file__))))

from dotenv import load_dotenv
from utils.asserts import assert_in_env
from os import getenv
from os.path import join

load_dotenv('.env')
# Configure deep sort info
deep_sort_info = dict(REID_CKPT=join(getenv('project_root'), getenv('reid_ckpt')),
                      MAX_DIST=0.2,
                      MIN_CONFIDENCE=.3,
                      NMS_MAX_OVERLAP=0.5,
                      MAX_IOU_DISTANCE=0.7,
                      N_INIT=3,
                      MAX_AGE=70,
                      NN_BUDGET=100)
deep_sort_dict = {'DEEPSORT': deep_sort_info}

# Configure yolov3 info

yolov3_info = dict(CFG=join(getenv('project_root'), getenv('yolov3_cfg')),
                   WEIGHT=join(getenv('project_root'), getenv('yolov3_weight')),
                   CLASS_NAMES=join(getenv('project_root'), getenv('yolov3_class_names')),
                   SCORE_THRESH=0.5,
                   NMS_THRESH=0.4
                   )
yolov3_dict = {'YOLOV3': yolov3_info}

# Configure yolov3-tiny info

yolov3_tiny_info = dict(CFG=join(getenv('project_root'), getenv('yolov3_tiny_cfg')),
                        WEIGHT=join(getenv('project_root'), getenv('yolov3_tiny_weight')),
                        CLASS_NAMES=join(getenv('project_root'), getenv('yolov3_class_names')),
                        SCORE_THRESH=0.5,
                        NMS_THRESH=0.4
                        )
yolov3_tiny_dict = {'YOLOV3': yolov3_tiny_info}


check_list = ['project_root', 'reid_ckpt', 'yolov3_class_names', 'model_type', 'yolov3_cfg', 'yolov3_weight',
              'yolov3_tiny_cfg', 'yolov3_tiny_weight', 'yolov3_class_names']

if assert_in_env(check_list):
    assert isfile(deep_sort_info['REID_CKPT'])
    if getenv('model_type') == 'yolov3':
        assert isfile(yolov3_info['WEIGHT'])
        assert isfile(yolov3_info['CFG'])
        assert isfile(yolov3_info['CLASS_NAMES'])
        model = yolov3_dict.copy()

    elif getenv('model_type') == 'yolov3_tiny':
        assert isfile(yolov3_tiny_info['WEIGHT'])
        assert isfile(yolov3_tiny_info['CFG'])
        assert isfile(yolov3_tiny_info['CLASS_NAMES'])
        model = yolov3_tiny_dict.copy()
    else:
        raise ValueError("Value '{}' for model_type is not valid".format(getenv('model_type')))

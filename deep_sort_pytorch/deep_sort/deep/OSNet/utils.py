from collections import OrderedDict
import pickle
from functools import partial
import warnings
import os.path as osp
from .OSNet import OSNet
import numpy as np
import zlib
import ast
import struct
import torch



def load_checkpoint(fpath: str):
    '''
    Brought from torchreid utils.py .
    More information: https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/utils/torchtools.html
    '''
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model: OSNet, weight_path: str):
    '''
    Brought from torchreid utils.py .
    More information: https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/utils/torchtools.html
    '''
    checkpoint = load_checkpoint(weight_path)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            '[OSNET info] The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path))
    else:
        print('[OSNET info] Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))


def compress_feature_vector(feature_vector: np.ndarray) -> str:
    '''
    :param feature_vector: (np.ndarray) the OSNet 512 dimensional feature vector.
    :return: (string) compressed feature vector
    '''
    # The input has to be a 1-dimensional ndarray
    vector = list(feature_vector)
    packed = struct.pack(
        f"{len(vector)}f",
        *vector
    )
    zlibed = zlib.compress(packed)
    return zlibed.hex()

def compress_bytes_image(cropped_image: bytes, image_width: int, image_height: int, colors: int) -> str:
    '''
    [INFO] -> This method is attached to this script to show how the image is compressed upstream.
    :param cropped_image: (bytes) a cropped bounding box containing the image of a person.
    :param image_width: (int) information about the bounding box width
    :param image_height: (int) information about the bounding box height
    :param colors: (int) number of channels of the bounding box
    :return: (str) the dictionary as a string
    '''
    # Debugging: information of the image size
    # print("The size of the input image is: ", sys.getsizeof(cropped_image), " bytes")
    zlibed = zlib.compress(cropped_image)
    # print("The size of the compressed image is: ", sys.getsizeof(zlibed), " bytes")
    img_dict = {
        "width": image_width,
        "height": image_height,
        "colors": colors,
        "image": zlibed.hex()
    }
    return img_dict.__str__()

def uncompress_string_image(compresed_cropped_image: str) -> bytes:
    '''
    [INFO] -> This method uncompresses the bytes image compressed as shown in the method "compress_bytes_image".
    :param compresed_cropped_image: (str) a dictionary as a string, that contains the crop information.
    :return: (bytes) A bytes image with the visual info about the detection.
    '''

    # Defensive programming: an empty field can be provided: If so, return an None value
    if compresed_cropped_image is not np.nan:
        compresed_dict = ast.literal_eval(compresed_cropped_image)
        # Debugging: information of the image size
        # print("The size of the compressed input image is: ", sys.getsizeof(compresed_cropped_image), " bytes")
        unhexed = bytes.fromhex(compresed_dict["image"])
        unzlibed = zlib.decompress(unhexed)
        patch_shape = (compresed_dict["height"], compresed_dict["width"], compresed_dict["colors"])
        # print("The size of the uncompressed image is: ", sys.getsizeof(unzlibed), " bytes")
        image_array = np.frombuffer(unzlibed, dtype='uint8').reshape(patch_shape)
        return image_array
    else:
        return None




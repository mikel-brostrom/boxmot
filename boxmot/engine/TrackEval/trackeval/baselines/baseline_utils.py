
import os
import csv
import numpy as np
from copy import deepcopy
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from trackeval.baselines.pascal_colormap import pascal_colormap


def load_seq(file_to_load):
    """ Load input data from file in RobMOTS format (e.g. provided detections).
    Returns: Data object with the following structure (see STP :
        data['cls'][t] = {'ids', 'scores', 'im_hs', 'im_ws', 'mask_rles'}
    """
    fp = open(file_to_load)
    dialect = csv.Sniffer().sniff(fp.readline(), delimiters=' ')
    dialect.skipinitialspace = True
    fp.seek(0)
    reader = csv.reader(fp, dialect)
    read_data = {}
    num_timesteps = 0
    for i, row in enumerate(reader):
        if row[-1] in '':
            row = row[:-1]
        t = int(row[0])
        cid = row[1]
        c = int(row[2])
        s = row[3]
        h = row[4]
        w = row[5]
        rle = row[6]

        if t >= num_timesteps:
            num_timesteps = t + 1

        if c in read_data.keys():
            if t in read_data[c].keys():
                read_data[c][t]['ids'].append(cid)
                read_data[c][t]['scores'].append(s)
                read_data[c][t]['im_hs'].append(h)
                read_data[c][t]['im_ws'].append(w)
                read_data[c][t]['mask_rles'].append(rle)
            else:
                read_data[c][t] = {}
                read_data[c][t]['ids'] = [cid]
                read_data[c][t]['scores'] = [s]
                read_data[c][t]['im_hs'] = [h]
                read_data[c][t]['im_ws'] = [w]
                read_data[c][t]['mask_rles'] = [rle]
        else:
            read_data[c] = {t: {}}
            read_data[c][t]['ids'] = [cid]
            read_data[c][t]['scores'] = [s]
            read_data[c][t]['im_hs'] = [h]
            read_data[c][t]['im_ws'] = [w]
            read_data[c][t]['mask_rles'] = [rle]
    fp.close()

    data = {}
    for c in read_data.keys():
        data[c] = [{} for _ in range(num_timesteps)]
        for t in range(num_timesteps):
            if t in read_data[c].keys():
                data[c][t]['ids'] = np.atleast_1d(read_data[c][t]['ids']).astype(int)
                data[c][t]['scores'] = np.atleast_1d(read_data[c][t]['scores']).astype(float)
                data[c][t]['im_hs'] = np.atleast_1d(read_data[c][t]['im_hs']).astype(int)
                data[c][t]['im_ws'] = np.atleast_1d(read_data[c][t]['im_ws']).astype(int)
                data[c][t]['mask_rles'] = np.atleast_1d(read_data[c][t]['mask_rles']).astype(str)
            else:
                data[c][t]['ids'] = np.empty(0).astype(int)
                data[c][t]['scores'] = np.empty(0).astype(float)
                data[c][t]['im_hs'] = np.empty(0).astype(int)
                data[c][t]['im_ws'] = np.empty(0).astype(int)
                data[c][t]['mask_rles'] = np.empty(0).astype(str)
    return data


def threshold(tdata, thresh):
    """ Removes detections below a certian threshold ('thresh') score. """
    new_data = {}
    to_keep = tdata['scores'] > thresh
    for field in ['ids', 'scores', 'im_hs', 'im_ws', 'mask_rles']:
        new_data[field] = tdata[field][to_keep]
    return new_data


def create_coco_mask(mask_rles, im_hs, im_ws):
    """ Converts mask as rle text (+ height and width) to encoded version used by pycocotools. """
    coco_masks = [{'size': [h, w], 'counts': m.encode(encoding='UTF-8')}
                  for h, w, m in zip(im_hs, im_ws, mask_rles)]
    return coco_masks


def mask_iou(mask_rles1, mask_rles2, im_hs, im_ws, do_ioa=0):
    """ Calculate mask IoU between two masks.
    Further allows 'intersection over area' instead of IoU (over the area of mask_rle1).
    Allows either to pass in 1 boolean for do_ioa for all mask_rles2 or also one for each mask_rles2.
    It is recommended that mask_rles1 is a detection and mask_rles2 is a groundtruth.
    """
    coco_masks1 = create_coco_mask(mask_rles1, im_hs, im_ws)
    coco_masks2 = create_coco_mask(mask_rles2, im_hs, im_ws)

    if not hasattr(do_ioa, "__len__"):
        do_ioa = [do_ioa]*len(coco_masks2)
    assert(len(coco_masks2) == len(do_ioa))
    if len(coco_masks1) == 0 or len(coco_masks2) == 0:
        iou = np.zeros(len(coco_masks1), len(coco_masks2))
    else:
        iou = mask_utils.iou(coco_masks1, coco_masks2, do_ioa)
    return iou


def sort_by_score(t_data):
    """ Sorts data by score """
    sort_index = np.argsort(t_data['scores'])[::-1]
    for k in t_data.keys():
        t_data[k] = t_data[k][sort_index]
    return t_data


def mask_NMS(t_data, nms_threshold=0.5, already_sorted=False):
    """ Remove redundant masks by performing non-maximum suppression (NMS) """

    # Sort by score
    if not already_sorted:
        t_data = sort_by_score(t_data)

    #  Calculate the mask IoU between all detections in the timestep.
    mask_ious_all = mask_iou(t_data['mask_rles'], t_data['mask_rles'], t_data['im_hs'], t_data['im_ws'])

    # Determine which masks NMS should remove
    # (those overlapping greater than nms_threshold with another mask that has a higher score)
    num_dets = len(t_data['mask_rles'])
    to_remove = [False for _ in range(num_dets)]
    for i in range(num_dets):
        if not to_remove[i]:
            for j in range(i + 1, num_dets):
                if mask_ious_all[i, j] > nms_threshold:
                    to_remove[j] = True

    # Remove detections which should be removed
    to_keep = np.logical_not(to_remove)
    for k in t_data.keys():
        t_data[k] = t_data[k][to_keep]

    return t_data


def non_overlap(t_data, already_sorted=False):
    """ Enforces masks to be non-overlapping in an image, does this by putting masks 'on top of one another',
    such that higher score masks 'occlude' and thus remove parts of lower scoring masks.

    Help wanted: if anyone knows a way to do this WITHOUT converting the RLE to the np.array let me know, because that
    would be MUCH more efficient. (I have tried, but haven't yet had success).
    """

    # Sort by score
    if not already_sorted:
        t_data = sort_by_score(t_data)

    # Get coco masks
    coco_masks = create_coco_mask(t_data['mask_rles'], t_data['im_hs'], t_data['im_ws'])

    # Create a single np.array to hold all of the non-overlapping mask
    masks_array = np.zeros((t_data['im_hs'][0], t_data['im_ws'][0]), 'uint8')

    # Decode each mask into a np.array, and place it into the overall array for the whole frame.
    # Since masks with the lowest score are placed first, they are 'partially overridden' by masks with a higher score
    # if they overlap.
    for i, mask in enumerate(coco_masks[::-1]):
        masks_array[mask_utils.decode(mask).astype('bool')] = i + 1

    # Encode the resulting np.array back into a set of coco_masks which are now non-overlapping.
    num_dets = len(coco_masks)
    for i, j in enumerate(range(1, num_dets + 1)[::-1]):
        coco_masks[i] = mask_utils.encode(np.asfortranarray(masks_array == j, dtype=np.uint8))

    # Convert from coco_mask back into our mask_rle format.
    t_data['mask_rles'] = [m['counts'].decode("utf-8") for m in coco_masks]

    return t_data


def masks2boxes(mask_rles, im_hs, im_ws):
    """ Extracts bounding boxes which surround a set of masks. """
    coco_masks = create_coco_mask(mask_rles, im_hs, im_ws)
    boxes = np.array([mask_utils.toBbox(x) for x in coco_masks])
    if len(boxes) == 0:
        boxes = np.empty((0, 4))
    return boxes


def box_iou(bboxes1, bboxes2, box_format='xywh', do_ioa=False, do_giou=False):
    """ Calculates the IOU (intersection over union) between two arrays of boxes.
    Allows variable box formats ('xywh' and 'x0y0x1y1').
    If do_ioa (intersection over area), then calculates the intersection over the area of boxes1 - this is commonly
    used to determine if detections are within crowd ignore region.
    If do_giou (generalized intersection over union, then calculates giou.
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        ious = np.zeros((len(bboxes1), len(bboxes2)))
        return ious
    if box_format in 'xywh':
        # layout: (x0, y0, w, h)
        bboxes1 = deepcopy(bboxes1)
        bboxes2 = deepcopy(bboxes2)

        bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
        bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
        bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
        bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
    elif box_format not in 'x0y0x1y1':
        raise (Exception('box_format %s is not implemented' % box_format))

    # layout: (x0, y0, x1, y1)
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

    if do_ioa:
        ioas = np.zeros_like(intersection)
        valid_mask = area1 > 0 + np.finfo('float').eps
        ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]

        return ioas
    else:
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
        intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
        intersection[union <= 0 + np.finfo('float').eps] = 0
        union[union <= 0 + np.finfo('float').eps] = 1
        ious = intersection / union

    if do_giou:
        enclosing_area = np.maximum(max_[..., 2] - min_[..., 0], 0) * np.maximum(max_[..., 3] - min_[..., 1], 0)
        eps = 1e-7
        # giou
        ious = ious - ((enclosing_area - union) / (enclosing_area + eps))

    return ious


def match(match_scores):
    match_rows, match_cols = linear_sum_assignment(-match_scores)
    return match_rows, match_cols


def write_seq(output_data, out_file):
    out_loc = os.path.dirname(out_file)
    if not os.path.exists(out_loc):
        os.makedirs(out_loc, exist_ok=True)
    fp = open(out_file, 'w', newline='')
    writer = csv.writer(fp, delimiter=' ')
    for row in output_data:
        writer.writerow(row)
    fp.close()


def combine_classes(data):
    """ Converts data from a class-separated to a class-combined format.
    Input format: data['cls'][t] = {'ids', 'scores', 'im_hs', 'im_ws', 'mask_rles'}
    Output format: data[t] = {'ids', 'scores', 'im_hs', 'im_ws', 'mask_rles', 'cls'}
    """
    output_data = [{} for _ in list(data.values())[0]]
    for cls, cls_data in data.items():
        for timestep, t_data in enumerate(cls_data):
            for k in t_data.keys():
                if k in output_data[timestep].keys():
                    output_data[timestep][k] += list(t_data[k])
                else:
                    output_data[timestep][k] = list(t_data[k])
            if 'cls' in output_data[timestep].keys():
                output_data[timestep]['cls'] += [cls]*len(output_data[timestep]['ids'])
            else:
                output_data[timestep]['cls'] = [cls]*len(output_data[timestep]['ids'])

    for timestep, t_data in enumerate(output_data):
        for k in t_data.keys():
            output_data[timestep][k] = np.array(output_data[timestep][k])

    return output_data


def save_as_png(t_data, out_file, im_h, im_w):
    """ Save a set of segmentation masks into a PNG format, the same as used for the DAVIS dataset."""

    if len(t_data['mask_rles']) > 0:
        coco_masks = create_coco_mask(t_data['mask_rles'], t_data['im_hs'], t_data['im_ws'])

        list_of_np_masks = [mask_utils.decode(mask) for mask in coco_masks]

        png = np.zeros((t_data['im_hs'][0], t_data['im_ws'][0]))
        for mask, c_id in zip(list_of_np_masks, t_data['ids']):
            png[mask.astype("bool")] = c_id + 1
    else:
        png = np.zeros((im_h, im_w))

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    colmap = (np.array(pascal_colormap) * 255).round().astype("uint8")
    palimage = Image.new('P', (16, 16))
    palimage.putpalette(colmap)
    im = Image.fromarray(np.squeeze(png.astype("uint8")))
    im2 = im.quantize(palette=palimage)
    im2.save(out_file)


def get_frame_size(data):
    """ Gets frame height and width from data. """
    for cls, cls_data in data.items():
        for timestep, t_data in enumerate(cls_data):
            if len(t_data['im_hs'] > 0):
                im_h = t_data['im_hs'][0]
                im_w = t_data['im_ws'][0]
                return im_h, im_w
    return None

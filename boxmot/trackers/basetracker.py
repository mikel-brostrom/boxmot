import numpy as np
import cv2 as cv
import hashlib
import colorsys


class BaseTracker(object):
    def __init__(self, det_thresh=0.3, max_age=30, min_hits=3, iou_threshold=0.3):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes

    def update(self, dets, img, embs=None):
        raise NotImplementedError("The update method needs to be implemented by the subclass.")

    def id_to_color(self, id, saturation=0.75, value=0.95):
        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        
        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]
        
        return bgr

    def plot_trajectory(self, img):

        thickness = 2
        fontscale = 0.5
        for a in self.active_tracks:
            
            if a.history_observations:
                
                o = a.history_observations[-1]
                img = cv.rectangle(
                    img,
                    (int(o[0]), int(o[1])),
                    (int(o[2]), int(o[3])),
                    self.id_to_color(a.id),
                    thickness
                )
                img = cv.putText(
                    img,
                    f'id: {int(a.id)}, conf: {a.conf:.2f}, c: {int(a.cls)}',
                    (int(o[0]), int(o[1]) - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    fontscale,
                    self.id_to_color(a.id),
                    thickness
                )
            if len(a.history_observations) > 3:
                for e, o in enumerate(a.history_observations):
                    trajectory_thickness = int(np.sqrt(float (e + 1)) * 1.2)
                    cv.circle(
                        img,
                        (int((o[0] + o[2]) / 2),
                        int((o[1] + o[3]) / 2)), 
                        2,
                        color=self.id_to_color(int(a.id)),
                        thickness=trajectory_thickness
                    )
        return img


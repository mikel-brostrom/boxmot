import os
import csv
import numpy as np
from ._base_dataset import _BaseDataset
from ..utils import TrackEvalException
from .. import utils
from .. import _timing


class DAVIS(_BaseDataset):
    """Dataset class for DAVIS tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/davis/davis_unsupervised_val/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/davis/davis_unsupervised_val/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'SPLIT_TO_EVAL': 'val',  # Valid: 'val', 'train'
            'CLASSES_TO_EVAL': ['general'],
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FILE': None,  # Specify seqmap file
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            # '{gt_folder}/Annotations_unsupervised/480p/{seq}'
            'MAX_DETECTIONS': 0  # Maximum number of allowed detections per sequence (0 for no threshold)
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        # defining a default class since there are no classes in DAVIS
        self.should_classes_combine = False
        self.use_super_categories = False

        self.gt_fol = self.config['GT_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']

        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']
        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.config['TRACKERS_FOLDER']

        self.max_det = self.config['MAX_DETECTIONS']

        # Get classes to eval
        self.valid_classes = ['general']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only general class is valid.')

        # Get sequences to eval
        if self.config["SEQ_INFO"]:
            self.seq_list = list(self.config["SEQ_INFO"].keys())
            self.seq_lengths = self.config["SEQ_INFO"]
        elif self.config["SEQMAP_FILE"]:
            self.seq_list = []
            seqmap_file = self.config["SEQMAP_FILE"]
            if not os.path.isfile(seqmap_file):
                raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if row[0] == '':
                        continue
                    seq = row[0]
                    self.seq_list.append(seq)
        else:
            self.seq_list = os.listdir(self.gt_fol)

        self.seq_lengths = {seq: len(os.listdir(os.path.join(self.gt_fol, seq))) for seq in self.seq_list}

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']
        for tracker in self.tracker_list:
            for seq in self.seq_list:
                curr_dir = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq)
                if not os.path.isdir(curr_dir):
                    print('Tracker directory not found: ' + curr_dir)
                    raise TrackEvalException('Tracker directory not found: ' +
                                             os.path.join(tracker, self.tracker_sub_fol, seq))
                tr_timesteps = len(os.listdir(curr_dir))
                if self.seq_lengths[seq] != tr_timesteps:
                    raise TrackEvalException('GT folder and tracker folder have a different number'
                                             'timesteps for tracker %s and sequence %s' % (tracker, seq))

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the DAVIS format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.
        [masks_void]: list of masks with void pixels (pixels to be ignored during evaluation)

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils
        from PIL import Image

        # File location
        if is_gt:
            seq_dir = os.path.join(self.gt_fol, seq)
        else:
            seq_dir = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq)

        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'dets', 'masks_void']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # read frames
        frames = [os.path.join(seq_dir, im_name) for im_name in sorted(os.listdir(seq_dir))]

        id_list = []
        for t in range(num_timesteps):
            frame = np.array(Image.open(frames[t]))
            if is_gt:
                void = frame == 255
                frame[void] = 0
                raw_data['masks_void'][t] = mask_utils.encode(np.asfortranarray(void.astype(np.uint8)))
            id_values = np.unique(frame)
            id_values = id_values[id_values != 0]
            id_list += list(id_values)
            tmp = np.ones((len(id_values), *frame.shape))
            tmp = tmp * id_values[:, None, None]
            masks = np.array(tmp == frame[None, ...]).astype(np.uint8)
            raw_data['dets'][t] = mask_utils.encode(np.array(np.transpose(masks, (1, 2, 0)), order='F'))
            raw_data['ids'][t] = id_values.astype(int)
        num_objects = len(np.unique(id_list))

        if not is_gt and num_objects > self.max_det > 0:
            raise Exception('Number of proposals (%i) for sequence %s exceeds number of maximum allowed proposals (%i).'
                            % (num_objects, seq, self.max_det))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data['mask_shape'] = np.array(Image.open(frames[0])).shape
        if is_gt:
            raw_data['num_gt_ids'] = num_objects
        else:
            raw_data['num_tracker_ids'] = num_objects
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detection masks.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        DAVIS:
            In DAVIS, the 4 preproc steps are as follow:
                1) There are no classes, all detections are evaluated jointly
                2) No matched tracker detections are removed.
                3) No unmatched tracker detections are removed.
                4) There are no ground truth detections (e.g. those of distractor classes) to be removed.
            Preprocessing special to DAVIS: Pixels which are marked as void in the ground truth are set to zero in the
                tracker detections since they are not considered during evaluation.
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        num_gt_dets = 0
        num_tracker_dets = 0
        unique_gt_ids = []
        unique_tracker_ids = []
        num_timesteps = raw_data['num_timesteps']

        # count detections
        for t in range(num_timesteps):
            num_gt_dets += len(raw_data['gt_dets'][t])
            num_tracker_dets += len(raw_data['tracker_dets'][t])
            unique_gt_ids += list(np.unique(raw_data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(raw_data['tracker_ids'][t]))

        data['gt_ids'] = raw_data['gt_ids']
        data['gt_dets'] = raw_data['gt_dets']
        data['similarity_scores'] = raw_data['similarity_scores']
        data['tracker_ids'] = raw_data['tracker_ids']

        # set void pixels in tracker detections to zero
        for t in range(num_timesteps):
            void_mask = raw_data['masks_void'][t]
            if mask_utils.area(void_mask) > 0:
                void_mask_ious = np.atleast_1d(mask_utils.iou(raw_data['tracker_dets'][t], [void_mask], [False]))
                if void_mask_ious.any():
                    rows, columns = np.where(void_mask_ious > 0)
                    for r in rows:
                        det = mask_utils.decode(raw_data['tracker_dets'][t][r])
                        void = mask_utils.decode(void_mask).astype(bool)
                        det[void] = 0
                        det = mask_utils.encode(np.array(det, order='F').astype(np.uint8))
                        raw_data['tracker_dets'][t][r] = det
        data['tracker_dets'] = raw_data['tracker_dets']

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = raw_data['num_tracker_ids']
        data['num_gt_ids'] = raw_data['num_gt_ids']
        data['mask_shape'] = raw_data['mask_shape']
        data['num_timesteps'] = num_timesteps
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_mask_ious(gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False)
        return similarity_scores

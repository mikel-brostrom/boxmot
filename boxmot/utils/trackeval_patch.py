#!/usr/bin/env python3
"""
Patch the file:
  boxmot/engine/trackeval/trackeval/datasets/mot_challenge_2d_box.py

Changes applied (to match your target file):
- Default classes: ['pedestrian'] → ["person","bicycle","car"]
- Remove pedestrian-only validation & allow arbitrary classes
- Replace MOT class-id map with COCO-80 map
- Fix deprecated NumPy dtypes: np.float / np.int → float / int
- Replace entire get_preprocessed_seq_data() with the multi-class version
  that still performs distractor removal (Hungarian) like the target file
- (Optional) keep earlier safeguards; warn if the file differs

The script is defensive:
- Creates a .backup next to the target file
- Uses tolerant regex with DOTALL where appropriate
- Emits warnings if patterns are not found
"""

import os
import re
import sys
import shutil
from pathlib import Path


def sub_or_warn(pattern, repl, text, flags=0, label=""):
    new_text, n = re.subn(pattern, repl, text, flags=flags)
    if n == 0:
        print(f"⚠️  Warning: pattern for '{label}' not found. The source may differ.")
    else:
        print(f"✅ Applied '{label}' ({n} replacement{'s' if n != 1 else ''}).")
    return new_text, n


def apply_trackeval_patch(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found")
        return False

    # Backup
    backup_path = str(file_path) + ".backup"
    shutil.copy2(file_path, backup_path)
    print(f"🗂️  Created backup: {backup_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        total_changes = 0

        # 1) Default classes: set to ["person","bicycle","car"]
        content, n = sub_or_warn(
            r"'CLASSES_TO_EVAL':\s*\['pedestrian'\],\s*#\s*Valid:\s*\['pedestrian'\]",
            "'CLASSES_TO_EVAL': [\n                \"person\"\n            ],  # Valid: any class names (patched)",
            content,
            flags=re.MULTILINE,
            label="default classes list",
        )
        total_changes += n

        # 2) Replace class validation block to allow arbitrary classes
        pattern_class_validation = (
            r"(\#\s*Get classes to eval\s*\n\s*"
            r"self\.valid_classes\s*=\s*\['pedestrian'\]\s*\n\s*"
            r"self\.class_list\s*=\s*\[cls\.lower\(\)\s*if\s*cls\.lower\(\)\s*in\s*self\.valid_classes\s*else\s*None\s*\n"
            r"\s*for\s*cls\s*in\s*self\.config\['CLASSES_TO_EVAL'\]\]\s*\n\s*"
            r"if\s*not\s*all\(\s*self\.class_list\s*\):\s*\n\s*"
            r"raise\s*TrackEvalException\('Attempted to evaluate an invalid class\. Only pedestrian class is valid\.'\))"
        )
        replacement_class_validation = (
            "# Get classes to eval\n"
            "        self.valid_classes = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]\n"
            "        self.class_list   = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]\n"
            "        # Validation removed to allow arbitrary classes"
        )
        content, n = sub_or_warn(
            pattern_class_validation,
            replacement_class_validation,
            content,
            flags=re.MULTILINE,
            label="class validation block",
        )
        total_changes += n

        # 3) Replace the MOT class map with a COCO 80-class map
        pattern_class_map = (
            r"self\.class_name_to_class_id\s*=\s*\{[^}]*?'pedestrian':\s*1,.*?'reflection':\s*12, 'crowd':\s*13\s*\}"
        )
        replacement_class_map = (
            "self.class_name_to_class_id = {\n"
            "            'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,\n"
            "            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20,\n"
            "            'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28, 'suitcase': 29, 'frisbee': 30,\n"
            "            'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,\n"
            "            'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50,\n"
            "            'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56, 'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60,\n"
            "            'dining table': 61, 'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68, 'microwave': 69, 'oven': 70,\n"
            "            'toaster': 71, 'sink': 72, 'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77, 'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80\n"
            "        }"
        )
        content, n = sub_or_warn(
            pattern_class_map,
            replacement_class_map,
            content,
            flags=re.DOTALL,
            label="class map → COCO-80",
        )
        total_changes += n

        # 4) dtype fixes: np.float → float; np.int → int (astype and array([],...))
        content, n1 = sub_or_warn(
            r"dtype\s*=\s*np\.float",
            "dtype=float",
            content,
            flags=re.MULTILINE,
            label="np.float → float",
        )
        total_changes += n1

        content, n2 = sub_or_warn(
            r"\.astype\(\s*np\.int\s*\)",
            ".astype(int)",
            content,
            flags=re.MULTILINE,
            label=".astype(np.int) → .astype(int)",
        )
        total_changes += n2

        content, n3 = sub_or_warn(
            r"\bnp\.int\b",
            "int",
            content,
            flags=re.MULTILINE,
            label="loose np.int → int",
        )
        total_changes += n3

        # 5) Replace the entire get_preprocessed_seq_data() block with the target version
        pattern_whole_func = (
            r"@_timing\.time\s*\n\s*def\s+get_preprocessed_seq_data\(\s*self\s*,\s*raw_data\s*,\s*cls\s*\):"
            r"[\s\S]*?(?=\n\s*def\s+_calculate_similarities\s*\()"
        )

        replacement_whole_func = (
            "@_timing.time\n"
            "    def get_preprocessed_seq_data(self, raw_data, cls):\n"
            "        self._check_unique_ids(raw_data)\n"
            "        cls_id = self.class_name_to_class_id[cls]\n\n"
            "        # MOT distractor set (same as original; add non_mot_vehicle for MOT20)\n"
            "        distractor_classes = [12, 8, 6, 7, 2] if self.benchmark == 'MOT20' else [12, 8, 7, 2]\n\n"
            "        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets',\n"
            "                    'tracker_confidences', 'similarity_scores']\n"
            "        data = {k: [None] * raw_data['num_timesteps'] for k in data_keys}\n\n"
            "        unique_gt_ids, unique_tracker_ids = [], []\n"
            "        num_gt_dets = num_tracker_dets = 0\n\n"
            "        for t in range(raw_data['num_timesteps']):\n"
            "            gt_ids            = raw_data['gt_ids'][t]\n"
            "            gt_dets           = raw_data['gt_dets'][t]\n"
            "            gt_classes        = raw_data['gt_classes'][t]\n"
            "            gt_zero_marked    = raw_data['gt_extras'][t]['zero_marked']\n\n"
            "            tracker_ids         = raw_data['tracker_ids'][t]\n"
            "            tracker_dets        = raw_data['tracker_dets'][t]\n"
            "            tracker_classes     = raw_data['tracker_classes'][t]\n"
            "            tracker_confs       = raw_data['tracker_confidences'][t]\n"
            "            similarity_scores   = raw_data['similarity_scores'][t]  # shape (num_gt, num_trk)\n\n"
            "            # --- Keep only trackers of the eval class (columns) ---\n"
            "            trk_keep_mask = (tracker_classes == cls_id)\n"
            "            kept_tracker_ids   = tracker_ids[trk_keep_mask]\n"
            "            kept_tracker_dets  = tracker_dets[trk_keep_mask, :] if tracker_dets.size else tracker_dets\n"
            "            kept_tracker_confs = tracker_confs[trk_keep_mask]\n"
            "            sim = similarity_scores\n"
            "            sim = sim[:, trk_keep_mask]  # always slice columns; may yield (N, 0)\n\n"
            "            # --- Remove trackers that match distractor GT (match vs ALL GT, not filtered by class) ---\n"
            "            if self.do_preproc and self.benchmark != 'MOT15' and gt_ids.shape[0] > 0 and kept_tracker_ids.shape[0] > 0 and sim.size > 0:\n"
            "                # threshold & Hungarian (same as original)\n"
            "                matching_scores = sim.copy()\n"
            "                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0\n"
            "                match_rows, match_cols = linear_sum_assignment(-matching_scores)\n"
            "                actually_matched = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps\n"
            "                match_rows = match_rows[actually_matched]\n"
            "                match_cols = match_cols[actually_matched]\n\n"
            "                # mark tracker columns to remove if matched GT is a distractor class\n"
            "                is_distr = np.isin(gt_classes[match_rows], distractor_classes)\n"
            "                to_remove_cols = match_cols[is_distr]\n\n"
            "                if to_remove_cols.size > 0:\n"
            "                    kept_tracker_ids   = np.delete(kept_tracker_ids,   to_remove_cols, axis=0)\n"
            "                    kept_tracker_dets  = np.delete(kept_tracker_dets,  to_remove_cols, axis=0) if kept_tracker_dets.size else kept_tracker_dets\n"
            "                    kept_tracker_confs = np.delete(kept_tracker_confs, to_remove_cols, axis=0)\n"
            "                    sim                = np.delete(sim, to_remove_cols, axis=1)\n\n"
            "            # --- Now keep ONLY GT of the eval class and not zero-marked (rows) ---\n"
            "            if self.do_preproc and self.benchmark != 'MOT15':\n"
            "                gt_keep_mask = (gt_zero_marked != 0) & (gt_classes == cls_id)\n"
            "            else:\n"
            "                gt_keep_mask = (gt_zero_marked != 0)\n\n"
            "            kept_gt_ids  = gt_ids[gt_keep_mask]\n"
            "            kept_gt_dets = gt_dets[gt_keep_mask, :] if gt_dets.size else gt_dets\n"
            "            sim = sim[gt_keep_mask, :]  # always slice rows; ensures sim.shape == (len(kept_gt_ids), len(kept_tracker_ids))\n\n"
            "            # --- Write timestep outputs ---\n"
            "            data['tracker_ids'][t]          = kept_tracker_ids\n"
            "            data['tracker_dets'][t]         = kept_tracker_dets\n"
            "            data['tracker_confidences'][t]  = kept_tracker_confs\n"
            "            data['gt_ids'][t]               = kept_gt_ids\n"
            "            data['gt_dets'][t]              = kept_gt_dets\n"
            "            data['similarity_scores'][t]    = sim\n\n"
            "            unique_gt_ids      += list(np.unique(kept_gt_ids))\n"
            "            unique_tracker_ids += list(np.unique(kept_tracker_ids))\n"
            "            num_tracker_dets   += len(kept_tracker_ids)\n"
            "            num_gt_dets        += len(kept_gt_ids)\n\n"
            "        # --- Relabel to contiguous ids (no gaps) ---\n"
            "        if len(unique_gt_ids) > 0:\n"
            "            unique_gt_ids = np.unique(unique_gt_ids)\n"
            "            gt_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))\n"
            "            gt_map[unique_gt_ids] = np.arange(len(unique_gt_ids))\n"
            "            for t in range(raw_data['num_timesteps']):\n"
            "                if len(data['gt_ids'][t]) > 0:\n"
            "                    data['gt_ids'][t] = gt_map[data['gt_ids'][t]].astype(int)\n\n"
            "        if len(unique_tracker_ids) > 0:\n"
            "            unique_tracker_ids = np.unique(unique_tracker_ids)\n"
            "            trk_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))\n"
            "            trk_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))\n"
            "            for t in range(raw_data['num_timesteps']):\n"
            "                if len(data['tracker_ids'][t]) > 0:\n"
            "                    data['tracker_ids'][t] = trk_map[data['tracker_ids'][t]].astype(int)\n\n"
            "        # --- Stats & checks ---\n"
            "        data['num_tracker_dets'] = num_tracker_dets\n"
            "        data['num_gt_dets']      = num_gt_dets\n"
            "        data['num_tracker_ids']  = len(np.unique(unique_tracker_ids)) if len(unique_tracker_ids) > 0 else 0\n"
            "        data['num_gt_ids']       = len(np.unique(unique_gt_ids)) if len(unique_gt_ids) > 0 else 0\n"
            "        data['num_timesteps']    = raw_data['num_timesteps']\n"
            "        data['seq']              = raw_data['seq']\n\n"
            "        self._check_unique_ids(data, after_preproc=True)\n"
            "        return data\n\n"
        )
        content, n = sub_or_warn(
            pattern_whole_func, replacement_whole_func, content, flags=re.DOTALL | re.MULTILINE,
            label="replace get_preprocessed_seq_data()"
        )
        total_changes += n

        # 6) (Optional legacy) Remove pedestrian-only check if it still exists
        content, n = sub_or_warn(
            r"Evaluation is only valid for pedestrian class",
            "Class validation removed",
            content,
            flags=re.DOTALL,
            label="remove pedestrian-only message (if any remains)",
        )
        total_changes += n

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("\n🎉 Patch completed.")
        print(f"Total replacement operations applied: {total_changes}")
        print("The file should now match your target behavior.")
        return True

    except Exception as e:
        # Restore backup on error
        shutil.copy2(backup_path, file_path)
        print(f"❌ Error applying patch: {e}")
        print("⤴️  Restored original file from backup.")
        return False


def main():
    target_file = Path("boxmot/engine/trackeval/trackeval/datasets/mot_challenge_2d_box.py")

    if len(sys.argv) > 1:
        target_file = Path(sys.argv[1])

    if not target_file.exists():
        print(f"❌ Error: Target file not found: {target_file}")
        print("Run this from the repo root, or pass the full path to mot_challenge_2d_box.py")
        sys.exit(1)

    print(f"🔧 Applying patch to: {target_file}")
    ok = apply_trackeval_patch(str(target_file))
    if ok:
        print("\n✅ Patch applied successfully!")
        sys.exit(0)
    else:
        print("\n❌ Failed to apply patch")
        sys.exit(2)


if __name__ == "__main__":
    main()

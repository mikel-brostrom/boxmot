#!/usr/bin/env python3
"""
Script to apply targeted patch modifications to:
  boxmot/engine/trackeval/trackeval/datasets/mot_challenge_2d_box.py

What this patch does:
- Change default classes from ['pedestrian'] to ['person', 'car']
- Remove "pedestrian-only" class validation and allow arbitrary classes
- Replace the MOT class-id map with a COCO 80-class map
- Update deprecated NumPy dtypes (np.float/np.int) → float/int
- Simplify distractor handling (empty list) and remove the MOT20 special-case append
- Keep all other logic intact

The script is defensive:
- Creates a .backup alongside the target file
- Uses tolerant regexes with DOTALL
- Warns if any expected pattern is not found (file may be a different version)
"""

import os
import re
import sys
import shutil
from pathlib import Path


def sub_or_warn(pattern, repl, text, flags=0, label=""):
    """Run re.sub with a counter; warn if no replacements were made."""
    new_text, n = re.subn(pattern, repl, text, flags=flags)
    if n == 0:
        print(f"⚠️  Warning: pattern for '{label}' not found. The source file may differ from the expected version.")
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

        # 1) Default classes ['pedestrian'] -> ['person', 'car']
        content, n = sub_or_warn(
            r"'CLASSES_TO_EVAL':\s*\['pedestrian'\],\s*#\s*Valid:\s*\['pedestrian'\]",
            "'CLASSES_TO_EVAL': ['person', 'car'],  # Valid: any class names (patched)",
            content,
            flags=re.MULTILINE,
            label="default classes",
        )
        total_changes += n

        # 2) Replace class validation logic block to allow arbitrary classes
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
        #    We anchor on presence of 'reflection' and 'crowd' keys to limit scope.
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

        # 4) dtype fixes: np.float → float; np.int → int in astype and np.array([], np.int)
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
            r"np\.array\(\s*\[\s*\]\s*,\s*np\.int\s*\)",
            "np.array([], int)",
            content,
            flags=re.MULTILINE,
            label="np.array([], np.int) → np.array([], int)",
        )
        total_changes += n3

        # 5) Remove specific distractor set list (make it empty)
        content, n = sub_or_warn(
            r"distractor_class_names\s*=\s*\[\s*'person_on_vehicle'\s*,\s*'static_person'\s*,\s*'distractor'\s*,\s*'reflection'\s*\]",
            "distractor_class_names = []",
            content,
            flags=re.MULTILINE,
            label="distractor list → []",
        )
        total_changes += n

        # 5b) Remove MOT20 special-case append for 'non_mot_vehicle' (prevents KeyError with COCO map)
        content, n = sub_or_warn(
            r"distractor_class_names\s*=\s*\[\]\s*\n\s*if\s+self\.benchmark\s*==\s*'MOT20':\s*\n\s*distractor_class_names\.append\('non_mot_vehicle'\)",
            "distractor_class_names = []  # no MOT20 distractors with COCO map",
            content,
            flags=re.MULTILINE,
            label="remove MOT20 distractor append",
        )
        total_changes += n

        # 6) Comment out pedestrian-only validation in get_preprocessed_seq_data
        pattern_ped_only_block = (
            r"(\#\s*Evaluation\s+is\s+ONLY\s+valid\s+for\s+pedestrian\s+class\s*\n\s*"
            r"if\s+len\(tracker_classes\)\s*>\s*0\s*and\s*np\.max\(tracker_classes\)\s*>\s*1:\s*\n\s*"
            r"raise\s*TrackEvalException\([\s\S]*?timestep %i\.'\s*%\s*\(np\.max\(tracker_classes\),\s*raw_data\['seq'\],\s*t\)\))"
        )
        replacement_ped_only_block = (
            "            # Class validation removed to allow arbitrary classes\n"
            "            # if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:\n"
            "            #     raise TrackEvalException(\n"
            "            #         'Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at '\n"
            "            #         'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))"
        )
        content, n = sub_or_warn(
            pattern_ped_only_block,
            replacement_ped_only_block,
            content,
            flags=re.DOTALL,
            label="remove pedestrian-only check",
        )
        total_changes += n

        # 7) Also fix any remaining 'np.int' occurrences that may appear elsewhere (safe)
        content, n = sub_or_warn(
            r"\bnp\.int\b",
            "int",
            content,
            flags=re.MULTILINE,
            label="loose np.int → int",
        )
        total_changes += n

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("\n🎉 Patch completed.")
        print(f"Total replacement operations applied: {total_changes}")
        print("The MotChallenge2DBox class should now support arbitrary object classes.")
        return True

    except Exception as e:
        # Restore backup on error
        shutil.copy2(backup_path, file_path)
        print(f"❌ Error applying patch: {e}")
        print("⤴️  Restored original file from backup.")
        return False


def main():
    # Default path as provided
    target_file = Path("boxmot/engine/trackeval/trackeval/datasets/mot_challenge_2d_box.py")

    # Allow optional CLI argument to override the path
    if len(sys.argv) > 1:
        target_file = Path(sys.argv[1])

    if not target_file.exists():
        print(f"❌ Error: Target file not found: {target_file}")
        print("Please ensure you're running this script from the correct directory,")
        print("and that the 'boxmot' directory structure exists.")
        sys.exit(1)

    print(f"🔧 Applying patch to: {target_file}")
    ok = apply_trackeval_patch(str(target_file))
    if ok:
        print("\n✅ Patch applied successfully!")
        print("Key changes made:")
        print("- Default classes changed from ['pedestrian'] to ['person', 'car']")
        print("- Class validation removed to allow arbitrary classes")
        print("- Class mapping expanded to 80 COCO classes")
        print("- Deprecated NumPy data types updated")
        print("- Distractor class handling simplified; MOT20 special-case removed")
        sys.exit(0)
    else:
        print("\n❌ Failed to apply patch")
        sys.exit(2)


if __name__ == "__main__":
    main()

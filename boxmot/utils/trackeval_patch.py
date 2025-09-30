#!/usr/bin/env python3  
"""  
Script to apply targeted patch modifications to boxmot/engine/trackeval/trackeval/datasets/mot_challenge_2d_box.py  
This script modifies only the specific parts that need to be changed.  
"""  
  
import os  
import re  
import shutil  
from pathlib import Path  
  
def apply_trackeval_patch(file_path):  
    """Apply only the necessary changes to the MOT Challenge dataset file"""  
      
    if not os.path.exists(file_path):  
        print(f"Error: File {file_path} not found")  
        return False  
      
    # Create backup  
    backup_path = str(file_path) + '.backup'  
    shutil.copy2(file_path, backup_path)  
    print(f"Created backup: {backup_path}")  
      
    try:  
        with open(file_path, 'r') as f:  
            content = f.read()  
          
        # 1. Change default classes configuration  
        content = re.sub(  
            r"'CLASSES_TO_EVAL': \['pedestrian'\],  # Valid: \['pedestrian'\]",  
            "'CLASSES_TO_EVAL': ['person', 'car'],  # Valid: any class names (patched)",  
            content  
        )  
          
        # 2. Replace class validation logic  
        old_class_validation = r"""        # Get classes to eval  
        self\.valid_classes = \['pedestrian'\]  
        self\.class_list = \[cls\.lower\(\) if cls\.lower\(\) in self\.valid_classes else None  
                           for cls in self\.config\['CLASSES_TO_EVAL'\]\]  
        if not all\(self\.class_list\):  
            raise TrackEvalException\('Attempted to evaluate an invalid class\. Only pedestrian class is valid\.'\)"""  
          
        new_class_validation = """        # Get classes to eval  
        self.valid_classes = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]    
        self.class_list = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]    
        # Validation removed to allow arbitrary classes"""  
          
        content = re.sub(old_class_validation, new_class_validation, content, flags=re.MULTILINE)  
          
        # 3. Replace class mapping with COCO classes  
        old_mapping = r"""        self\.class_name_to_class_id = \{'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,  
                                       'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,  
                                       'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13\}"""  
          
        new_mapping = """        self.class_name_to_class_id = {  
            'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,  
            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20,  
            'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28, 'suitcase': 29, 'frisbee': 30,  
            'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,  
            'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50,  
            'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56, 'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60,  
            'dining table': 61, 'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68, 'microwave': 69, 'oven': 70,  
            'toaster': 71, 'sink': 72, 'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77, 'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80  
        }"""  
          
        content = re.sub(old_mapping, new_mapping, content, flags=re.MULTILINE)  
          
        # 4. Fix deprecated NumPy data types  
        content = re.sub(r'dtype=np\.float', 'dtype=float', content)  
        content = re.sub(r'np\.array\(\[\], np\.int\)', 'np.array([], int)', content)  
        content = re.sub(r'\.astype\(np\.int\)', '.astype(int)', content)  
          
        # 5. Remove distractor classes  
        content = re.sub(  
            r"distractor_class_names = \['person_on_vehicle', 'static_person', 'distractor', 'reflection'\]",  
            "distractor_class_names = []",  
            content  
        )  
          
        # 6. Comment out pedestrian-only validation  
        old_validation = r"""            # Evaluation is ONLY valid for pedestrian class  
            if len\(tracker_classes\) > 0 and np\.max\(tracker_classes\) > 1:  
                raise TrackEvalException\(  
                    'Evaluation is only valid for pedestrian class\. Non pedestrian class \(%i\) found in sequence %s at '  
                    'timestep %i\.' % \(np\.max\(tracker_classes\), raw_data\['seq'\], t\)\)"""  
          
        new_validation = """            # Class validation removed to allow arbitrary classes    
            # if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:    
            #     raise TrackEvalException(    
            #         'Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at '    
            #         'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))"""  
          
        content = re.sub(old_validation, new_validation, content, flags=re.MULTILINE)  
          
        # Write the modified content back  
        with open(file_path, 'w') as f:  
            f.write(content)  
          
        print(f"Successfully applied targeted patch to {file_path}")  
        return True  
          
    except Exception as e:  
        # Restore backup on error  
        shutil.copy2(backup_path, file_path)  
        print(f"Error applying patch: {e}")  
        print(f"Restored original file from backup")  
        return False  
  
def main():  
    """Main function to apply the patch"""  
      
    # Use the specific path provided by the user  
    target_file = Path("boxmot/engine/trackeval/trackeval/datasets/mot_challenge_2d_box.py")  
      
    if not target_file.exists():  
        print(f"Error: Target file not found: {target_file}")  
        print("Please ensure you're running this script from the correct directory")  
        print("and that the boxmot directory structure exists.")  
        return  
      
    print(f"Applying patch to: {target_file}")  
      
    success = apply_trackeval_patch(str(target_file))  
      
    if success:  
        print("\n✅ Patch applied successfully!")  
        print("The MotChallenge2DBox class now supports arbitrary object classes.")  
        print("\nKey changes made:")  
        print("- Default classes changed from ['pedestrian'] to ['person', 'car']")  
        print("- Class validation removed to allow arbitrary classes")  
        print("- Class mapping expanded to 80 COCO classes")  
        print("- Deprecated NumPy data types updated")  
        print("- Distractor class handling simplified")  
        print("- Pedestrian-only validation commented out")  
    else:  
        print("\n❌ Failed to apply patch")  
  
if __name__ == "__main__":  
    main()
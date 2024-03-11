import numpy as np

# Sample detections for three different classes
# Assume each row is a detection with format: [x1, y1, x2, y2, confidence, class_id]
detections_class_1 = np.array([
    [10, 20, 30, 40, 0.95, 1],
    [15, 25, 35, 45, 0.85, 1]
])

detections_class_2 = np.array([
    [50, 60, 70, 80, 0.90, 2]
])

detections_class_3 = np.array([
    [20, 30, 40, 50, 0.88, 3],
    [25, 35, 45, 55, 0.92, 3],
    [30, 40, 50, 60, 0.81, 3]
])

# Collect these arrays in a list
detections_list = [detections_class_1, detections_class_2, detections_class_3]

# Use np.vstack() to stack these arrays vertically into a single array
all_detections = np.vstack(detections_list)

print("Combined Detections:\n", all_detections)
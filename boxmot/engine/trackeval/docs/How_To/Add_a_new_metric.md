# How to add a new or custom family of evaluation metrics to TrackEval

 - Create your metrics code in ```trackeval/metrics/<your_metric>.py```.
 - It's probably easiest to start by copying an existing metrics code and editing it, e.g. ```trackeval/metrics/identity.py``` is probably the simplest.
 - Your metric should be class, and it should inherit from the ```trackeval.metrics._base_metric._BaseMetric``` class.
 - Define an ```__init__``` function that defines the different ```fields``` (values) that your metric will calculate. See ```trackeval/metrics/_base_metric.py``` for a list of currently used field types. Feel free to add new types.
 - Define your code to actually calculate your metric for a single sequence and single class in a function called ```eval_sequence```, which takes a data dictionary as input, and returns a results dictionary as output.
 - Define functions for how to combine your metric field values over a) sequences ```combine_sequences```, b) over classes ```combine_classes_class_averaged```, and c) over classes weighted by the number of detections ```combine_classes_det_averaged```.
 - We find using a function such as the ```_compute_final_fields``` function that we use in the current metrics is convienient because it is likely used for metrics calculation and for the different metric combination, however this is not required.
 - Register your new metric by adding it to ```trackeval/metrics/init.py```  
 - Your new metric can be used by passing the metrics class to a list of metrics which is passed to the evaluator (see files in ```scripts/*```).

from boxmot.utils.checks import TestRequirements

tr = TestRequirements()


def get_yolo_inferer(yolo_model):

    if 'yolox' in str(yolo_model):
        try:
            import yolox  # for linear_assignment
            assert yolox.__version__
        except (ImportError, AssertionError, AttributeError):
            tr.check_packages(('yolox==0.3.0',), cmds='--no-dependencies')
            tr.check_packages(('tabulate',))  # needed dependency
            tr.check_packages(('thop',))  # needed dependency
        from .yolox import YoloXStrategy
        return YoloXStrategy
    elif 'yolov8' in str(yolo_model):
        # ultralytics already installed when running track.py
        from .yolov8 import Yolov8Strategy
        return Yolov8Strategy
    elif 'yolo_nas' in str(yolo_model):
        try:
            import super_gradients  # for linear_assignment
            assert super_gradients.__version__
        except (ImportError, AssertionError, AttributeError):
            tr.check_packages(('super-gradients==3.1.1',))  # install
        from .yolonas import YoloNASStrategy
        return YoloNASStrategy

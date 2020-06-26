"""

# TODO: Load ML model with redis and keep it for sometime.
    1- detector/yolov3/detector.py |=> yolov3 weightfile -> redis cache
    2- deepsort/deep/feature_extractor |=> model_path -> redis cache
    3- Use tmpfs (Insert RAM as a virtual disk and store model state): https://pypi.org/project/memory-tempfile/

"""
from os.path import join
from os import getenv, environ
from dotenv import load_dotenv
import argparse
from threading import Thread

from redis import Redis
from flask import Response, Flask, jsonify, request, abort

from rtsp_threaded_tracker import RealTimeTracking
from server_cfg import model, deep_sort_dict
from config.config import DevelopmentConfig
from utils.parser import get_config

redis_cache = Redis('127.0.0.1')
app = Flask(__name__)
environ['in_progress'] = 'off'


def parse_args():
    """
    Parses the arguments
    Returns:
        argparse Namespace
    """
    assert 'project_root' in environ.keys()
    project_root = getenv('project_root')
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type=str,
                        default=getenv('camera_stream'))

    parser.add_argument("--model",
                        type=str,
                        default=join(project_root,
                                     getenv('model_type')))

    parser.add_argument("--cpu",
                        dest="use_cuda",
                        action="store_false", default=True)
    args = parser.parse_args()

    return args


def gen():
    """

    Returns: video frames from redis cache

    """
    while True:
        frame = redis_cache.get('frame')
        if frame is not None:
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


def pedestrian_tracking(cfg, args):
    """
    starts the pedestrian detection on rtsp link
    Args:
        cfg:
        args:

    Returns:

    """
    tracker = RealTimeTracking(cfg, args)
    tracker.run()


def trigger_process(cfg, args):
    """
    triggers pedestrian_tracking process on rtsp link using a thread
    Args:
        cfg:
        args:

    Returns:
    """
    try:
        t = Thread(target=pedestrian_tracking, args=(cfg, args))
        t.start()
        return jsonify({"message": "Pedestrian detection started successfully"})
    except Exception:
        return jsonify({'message': "Unexpected exception occured in process"})


@app.errorhandler(400)
def bad_argument(error):
    return jsonify({'message': error.description['message']})


# Routes
@app.route('/stream', methods=['GET'])
def stream():
    """
    Provides video frames on http link
    Returns:

    """
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/run", methods=['GET'])
def process_manager():
    """
    request parameters:
    run (bool): 1  -> start the pedestrian tracking
                0  -> stop it
    camera_stream: str -> rtsp link to security camera

    :return:
    """
    # data = request.args
    data = request.args
    status = data['run']
    status = int(status) if status.isnumeric() else abort(400, {'message': f"bad argument for run {data['run']}"})
    if status == 1:
        # if pedestrian tracking is not running, start it off!
        try:
            if environ.get('in_progress', 'off') == 'off':
                global cfg, args
                vdo = data.get('camera_stream')
                if vdo is not None:
                    args.input = int(vdo)
                environ['in_progress'] = 'on'
                return trigger_process(cfg, args)
            elif environ.get('in_progress') == 'on':
                # if pedestrian tracking is running, don't start another one (we are short of gpu resources)
                return jsonify({"message": " Pedestrian detection is already in progress."})
        except Exception:
            environ['in_progress'] = 'off'
            return abort(503)
    elif status == 0:
        if environ.get('in_progress', 'off') == 'off':
            return jsonify({"message": "pedestrian detection is already terminated!"})
        else:
            environ['in_progress'] = 'off'
            return jsonify({"message": "Pedestrian detection terminated!"})


if __name__ == '__main__':
    load_dotenv()
    app.config.from_object(DevelopmentConfig)

    # BackProcess Initialization
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_dict(model)
    cfg.merge_from_dict(deep_sort_dict)
    # Start the flask app
    app.run()

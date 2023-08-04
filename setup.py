# AGPL-3.0 license

import re
from pathlib import Path

import pkg_resources as pkg
from setuptools import find_packages, setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]


def get_version():
    file = PARENT / 'boxmot/__init__.py'
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', file.read_text(encoding='utf-8')).group(1)
    return version


setup(
    name='boxmot',  # name of pypi package
    version=get_version(),  # version of pypi package
    python_requires='>=3.8',
    license='AGPL-3.0',
    description=('SOTA tracking methods for detection, segmentation and pose estimation models.'),
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/mikel-brostrom/yolov8_tracking',
    project_urls={
        'Bug Reports': 'https://github.com/mikel-brostrom/yolo_tracking/issues',
        'Source': 'https://github.com/mikel-brostrom/yolo_tracking'},
    author='Mikel Brostrom',
    author_email='yolov5.deepsort.pytorch@gmail.com',
    packages=find_packages(),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'coverage',
            'pre-commit'
        ],
        'export': [
            'onnx>=1.12.0  ',  # ONNX export
            'onnxsim>=0.4.1 ',  # ONNX simplifier
            'nvidia-pyindex',  # TensorRT export
            'nvidia-tensorrt',  # TensorRT export
            'openvino-dev>=2022.3',  # OpenVINO export
            'onnx2tf>=1.10.0',  # TFLite export
            'onnx_graphsurgeon',  # TFLite export
            'sng4onnx',  # TFLite export
        ],
        'evolve': [
            'optuna',  # ONNX export
            'plotly',  # ONNX simplifier
            'kaleido',  # TensorRT export
            'joblib',  # TensorRT export
        ],
        'yolo': [
            'yolox==0.3.0',  # yolox inference
            'thop',  # yolox dependency
            'super-gradients==3.1.1',  # yolo_nas inference
            'ultralytics==8.0.124',  # Tyolov8 inference
        ],
    },

    platforms=["linux", "windows"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    keywords='machine-learning, deep-learning, vision, ML, DL, AI, YOLO',
)

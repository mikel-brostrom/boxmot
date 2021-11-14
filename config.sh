#!/bin/sh

# YOLOV5実行時の設定を記載
YOLOV5_BATCH_SIZE=8
YOLOV5_EPOCHS=40
YOLOV5_WEIGHTS=yolov5s.pt

# 訓練済みのモデルで物体追跡をする場合はここを1に設定
JUST_PREDICTION=0

# YOLOV5の訓練に自前のデータセットを使用する場合はここを1に設定
USE_CUSTOM_DATASET=0

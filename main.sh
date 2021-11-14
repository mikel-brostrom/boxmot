#!/bin/sh

if [ $JUST_PREDICTION -eq 1 ]; then
  # YOLOV5の学習済みモデルの重みをダウンロード
  mkdir -p /app/yolov5/runs/train/exp/weights
  wget -P /app/yolov5/runs/train/exp/weights /app/data -nc https://tf-analytics-reference.s3.ap-northeast-1.amazonaws.com/yolov5-deepsort-pytorch/best.pt
else
  if [ $USE_CUSTOM_DATASET -eq 0 ]; then
    # yolov5学習用データセットを取得
    wget -P /app/data -nc https://tf-analytics-reference.s3.ap-northeast-1.amazonaws.com/yolov5-deepsort-pytorch/train.zip
    wget -P /app/data -nc https://tf-analytics-reference.s3.ap-northeast-1.amazonaws.com/yolov5-deepsort-pytorch/val.zip
    wget -P /app/data -nc https://tf-analytics-reference.s3.ap-northeast-1.amazonaws.com/yolov5-deepsort-pytorch/test.zip
    wget -P /app/yolov5/data -nc https://tf-analytics-reference.s3.ap-northeast-1.amazonaws.com/yolov5-deepsort-pytorch/data.yaml
    unzip -o /app/data/train.zip -d /app/yolov5/data/
    unzip -o /app/data/val.zip -d /app/yolov5/data/
    unzip -o /app/data/test.zip -d /app/data/
  fi
  
  # yolov5の学習を実行
  cd /app/yolov5
  python train.py --img 640 --batch ${YOLOV5_BATCH_SIZE} --epochs ${YOLOV5_EPOCHS} --data data/data.yaml --weights ${YOLOV5_WEIGHTS}
fi


# 学習済みyolov5モデルを使用し、deepsortによるオブジェクトトラッキングを実行
cd /app
python track.py --source data/test/test.mp4 --save-vid --save-txt --yolo_weights yolov5/runs/train/exp/weights/best.pt
cp -r inference/output/* data/result
cp yolov5/runs/train/exp/weights/best.pt data/result/

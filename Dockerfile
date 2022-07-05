FROM ultralytics/yolov5:v6.1
WORKDIR /app
ADD ./requirements.txt .

RUN pip install -r requirements.txt

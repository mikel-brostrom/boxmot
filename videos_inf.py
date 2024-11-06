import cv2

# 读取视频文件
# video_path = r'E:\Huayi\monit_video_data\strawberryVideo_20221218_v040_l23\L3_2\RGB.mp4'
video_path = r'/home/xplv/huanghanyang/Track_Datasets/train/strawberryVideo_20222023testDS_v040_L3_2.mp4'
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
else:
    # 获取视频的分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频的帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0

    # 输出视频信息
    print(f"视频分辨率: {width}x{height}")
    print(f"帧数: {frame_count}")
    print(f"帧率: {fps} FPS")
    print(f"时长: {duration:.2f} 秒")

# 释放视频对象
cap.release()

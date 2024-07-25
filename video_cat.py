
import cv2

# 入力動画のパス
input_video_path = '/home/wada_docker/Documents/M2/boxmot/assets/A_1_Swim.mp4'
# 出力動画のパス
output_video_path = 'output_cat_video.mp4'

# カットする割合（例えば、幅と高さの10%をカットする場合）
cut_percentage = 0.2

# 動画の読み込み
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# カットする領域の幅と高さを割合で計算
cut_width = int(frame_width * (1 - 2 * cut_percentage))
cut_height = int(frame_height * (1 - 2 * cut_percentage))

# 中心の座標
center_x = frame_width // 2
center_y = frame_height // 2

# カットする領域の左上の座標
x_start = center_x - cut_width // 2
y_start = center_y - cut_height // 2

# 出力動画の設定
out = cv2.VideoWriter(output_video_path, fourcc, fps, (cut_width, cut_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 指定した割合で領域を切り出し
    cut_frame = frame[y_start:y_start + cut_height, x_start:x_start + cut_width]
    # 出力動画に書き込み
    out.write(cut_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

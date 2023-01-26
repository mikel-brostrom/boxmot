import os
import argparse

import cv2
import pandas as pd

def read_txt(path):
    df = pd.read_table(path, sep=' ', header=None)
    df = df.iloc[:, 0:6]
    df = df.set_axis(['frame', 'id', 'left', 'top', 'w', 'h'], axis=1)
    return df

def is_overlap_frame(startframe1, endframe1, startframe2, endframe2):
    return startframe1 <= endframe2 and endframe1 >= startframe2

def get_earliest_frame(*frames):
    return min(frames)

def get_latest_frame(*frames):
    return max(frames)

def extract_frames(df):
    frames = []
    for i in df.id.unique():
        mean = df[df.id == i].left.diff().fillna(0).mean()
        lmax = df[df.id == i].left.max()
        lmin = df[df.id == i].left.min()
        if (lmax - lmin > 3500) and mean >= 15:
            data = df[df.id == i]
            min_frame = data[data.left == lmin].frame.values[0]
            max_frame = data[data.left == lmax].frame.values[0]
            if min_frame > max_frame: continue

            if len(frames) == 0:
                frames.append([min_frame, max_frame])
                continue
            if not is_overlap_frame(min_frame, max_frame, frames[-1][0], frames[-1][1]):
                frames.append([min_frame, max_frame])
            else:
                frames[-1][0] = get_earliest_frame(frames[-1][0], min_frame)
                frames[-1][1] = get_latest_frame(frames[-1][1], max_frame)
    frames.sort()
    return frames

def trim_video(frames, input, output):
    video = cv2.VideoCapture(input)
    fps = video.get(cv2.CAP_PROP_FPS)
    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps, w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if not os.path.exists(output):
        os.makedirs(output)

    idx = 0
    flag = 0
    frame_num = 0
    while True:
        ret, frame = video.read()
        if ret:
            if frames[idx][1] - frames[idx][0] > 1200: idx += 1
            if flag == 0 and frame_num >= frames[idx][0]:
                output_path = output + f'trimed_{frame_num}.mp4'
                print(output_path)
                writer = cv2.VideoWriter(output_path, fourcc, int(fps), (int(w), int(h)))
                flag = 1
            if flag == 1 and frame_num > frames[idx][1]:
                flag = 0
                idx += 1
                writer.release()
            if idx == len(frames): break
            if flag == 1: writer.write(frame)
            frame_num += 1
        else: break
    video.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="input video path")
    parser.add_argument("--output_dir", type=str, required=True, help="output video dir")
    parser.add_argument("--txt_path", type=str, required=True, help="tracking result txt path")
    args = parser.parse_args()

    track_data = read_txt(args.txt_path)
    frames = extract_frames(track_data)
    trim_video(frames, args.input_path, args.output_dir)


if __name__ == '__main__':
    main()

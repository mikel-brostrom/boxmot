import os
import csv
from pathlib import Path


# 读取文件并统计唯一ID数量
def read_file(file_path):
    global track_id, total_count, class_counts, track_id_set
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split(',')  # 假设数据是逗号分隔
            object_id = data[2]
            class_name = data[1]
            if object_id not in track_id_set:
                track_id_set.add(object_id)  # 将track_id加入集合
                total_count += 1  # 更新总数量
                # 更新每个类别的数量
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
            # if data[1] == 'Flower_':
            #     continue
    return track_id_set  # 返回唯一ID的集合


def save_statistics_to_txt(txt_file):
    """保存统计信息到txt文件"""
    with open(txt_file, "w") as f:
        f.write(f"总果实数量: {total_count}\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}:{count}\n")


def print_fruit_statistics():
    global total_count, class_counts
    print(f"总果实数量: {total_count}")
    for class_name, count in class_counts.items():
        print(f"类别 '{class_name}' 的数量: {count}")


if __name__ == "__main__":
    total_count = 0  # 总果实数量
    class_counts = {
        "Unripe_": 0,
        "Ripe_": 0,
        "Ripe7_": 0,
        "Ripe4_": 0,
        "Ripe2_": 0,
        "Flower_": 0,
        "Disease_": 0
    }
    track_id_set = set()  # 用于记录已统计的track_id
    save_name = '_track_results_strong_resnet.txt'
    # directory_path = r"D:\华毅\目标追踪数据集\1_艾维"  # 文件所在的目录
    file_path = r'/home/xplv/fenghao/Video_Strawberry_Screenshot/L3_2.txt'
    file_set = read_file(file_path)
    print_fruit_statistics()
    source_path = Path(file_path)
    source_dir = source_path.parent
    source_name = source_path.stem
    result_file = source_dir / f"{source_name}_gt_result.txt"
    save_statistics_to_txt(result_file)


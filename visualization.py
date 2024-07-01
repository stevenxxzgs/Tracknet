import pandas as pd
import math
import cv2
import numpy as np

# 视频文件路径
video_path = '/ssd2/cz/Lindan/game9/clip1/clip1.mp4'
# 假设CSV文件的路径
key_csv_path = '/ssd2/cz/Lindan/game9/clip1/Label.csv'
csv_path = '/ssd2/cz/TrackNetV3/pred_result/clip1_ball.csv'
# 读取CSV文件
key_df = pd.read_csv(key_csv_path)
df = pd.read_csv(csv_path)


frame_count = 0  # 当前帧计数器


key_x, key_y = key_df.iloc[0:, 2], key_df.iloc[0:, 3]
x, y = df.iloc[0:, 2], df.iloc[0:, 3]

# 打开视频文件
cap = cv2.VideoCapture(video_path)
# 检查是否成功打开视频文件
if not cap.isOpened():
    print("Error opening video file")


output_path = 'compare.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码方式，取决于输出格式
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # 在小球位置画一个红色的点
    cv2.circle(frame, (key_x[frame_count], key_y[frame_count]), 10, (0, 0, 255), -1)
    cv2.circle(frame, (x[frame_count], y[frame_count]), 10, (0, 255, 255), -1)
    # 显示帧
    # cv2.imshow('Video with Ball Marker', frame)
    out.write(frame)
    # 等待按键，如果按下 'q' 则退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# 清理
cap.release()
out.release()
cv2.destroyAllWindows()
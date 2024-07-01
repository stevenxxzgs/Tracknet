import pandas as pd
import math
# 假设CSV文件的路径
key_csv_path = '/ssd2/cz/Lindan/game2/clip1/Label.csv'
csv_path = '/ssd2/cz/TrackNetV3/pred_result/clip1_ball.csv'

# 读取CSV文件
key_df = pd.read_csv(key_csv_path)
df = pd.read_csv(csv_path)

def get_acc():
    key_ball_exist = key_df.iloc[0:, 1]
    ball_exist = df.iloc[0:, 1]
    # 初始化总帧数
    key_item = 0
    # 初始化和key存在球一致的帧数
    item = 0
    for i in range(len(key_ball_exist)):
        key_item += 1
        if key_ball_exist[i] == 1 and ball_exist[i] == 1:
            item += 1
        if key_ball_exist[i] == 0 and ball_exist[i] == 0:
            item += 1
    return float(item/key_item)


def get_precision():
    key_x, key_y = key_df.iloc[0:, 2], key_df.iloc[0:, 3]
    x, y = df.iloc[0:, 2], df.iloc[0:, 3]
    # 初始化总帧数
    key_item = 0
    # 初始化和key距离接近的帧数
    item = 0
    for i in range(len(key_x)):
        key_item += 1
        if math.sqrt((key_x[i] -x[i]) ** 2 + (key_y[i] -y[i]) ** 2) < 50:
            item += 1
    return float(item/key_item)

print("准确率", get_acc())
print("精确率", get_precision())

import os
import cv2
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from dataset import Badminton_Dataset
from utils import *
import sys, getopt
from utils_for_imgLabel import save_info, load_info, go2frame, show_image
# from torch.nn import DataParallel
# python3 easylabel.py --video_file pred_result/lindan2.mp4 --threshold 5

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str,default="pred_result/lindan2.mp4")
parser.add_argument('--model_file', type=str, default='exp/bestbest.pt')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--threshold', type=float, default=6.0)
parser.add_argument('--start_width', type=int, default=540)
parser.add_argument('--end_width', type=int, default=1230)
parser.add_argument('--save_dir', type=str, default='pred_result')
args = parser.parse_args()
# 推理参数
video_file = args.video_file
model_file = args.model_file
num_frame = args.num_frame
batch_size = args.batch_size
save_dir = args.save_dir
threshold = args.threshold
start_width = args.start_width
end_width = args.end_width
video_name = video_file.split('/')[-1][:-4]
out_csv_file = f'{save_dir}/{video_name}_ball{threshold}.csv'
# 打标参数
csv_path = out_csv_file
video_path = video_file


checkpoint = torch.load(model_file)
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
 
# Load model
model = get_model(model_name, num_frame, input_type)
model.load_state_dict(checkpoint['model_state_dict'])
model = torch.nn.DataParallel(model, device_ids =[0,1])
model = model.cuda()

model.eval()

conv_kernel = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
conv_kernel.weight.data.fill_(1)
conv_kernel  = torch.nn.DataParallel(conv_kernel, device_ids =[0,1])
conv_kernel = conv_kernel.cuda()



# Write csv file head
f = open(out_csv_file, 'w')
f.write('Frame,Visibility,X,Y\n')

# Cap configuration
cap = cv2.VideoCapture(video_file)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
success = True
frame_count = 0
num_final_frame = 0
ratio_h = h / HEIGHT
ratio_w = w / WIDTH

while success:
    print(f'Number of sampled frames: {frame_count}')
    # Sample frames to form input sequence
    frame_queue = []
    for _ in range(num_frame*batch_size):
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            frame_queue.append(frame)

    if not frame_queue:
        break
    
    # If mini batch incomplete
    if len(frame_queue) % num_frame != 0:
        frame_queue = []
        # Record the length of remain frames
        num_final_frame = len(frame_queue) +1
        print(num_final_frame)
        # Adjust the sample timestampe of cap
        frame_count = frame_count - num_frame*batch_size
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        # Re-sample mini batch
        for _ in range(num_frame*batch_size):
            success, frame = cap.read()
            if not success:
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_count += 1
                frame_queue.append(frame)
        if len(frame_queue) % num_frame != 0:
            continue
    
    x = get_frame_unit(frame_queue, num_frame)
    # Inference
    with torch.no_grad():
        y_pred = model(x.cuda())
        output_tensor = conv_kernel(y_pred.to(dtype=torch.float32)) #求和卷积层
    
    for i in range(y_pred.shape[1]):
        if num_final_frame > 0 and i < (num_frame*batch_size - num_final_frame-1):
            print('aaa')
            # Special case of last incomplete mini batch
            # Igore the frame which is already written to the output video
            continue 
        else:

            start_width_in_output_tensor = int(start_width/ratio_w)  # 开始的宽度索引
            end_width_in_output_tensor = int(end_width/ratio_w)    # 结束的宽度索引
            # print("output_tensor[0, i, :, :].shape",output_tensor[0, i, :, start_width_in_output_tensor:end_width_in_output_tensor].shape)
            # 找出输出特征层中的最大值及其位置
            max_index = torch.argmax(output_tensor[0, i, :,start_width_in_output_tensor:end_width_in_output_tensor])
            # 将最大值位置转换为 (x, y) 坐标
            max_2d_index = np.unravel_index(max_index.item(), output_tensor[0, i, :, start_width_in_output_tensor:end_width_in_output_tensor].shape)
            if output_tensor[0, i, max_2d_index[0], max_2d_index[1]+start_width_in_output_tensor] > threshold:
                cx_pred, cy_pred = int(ratio_w*(max_2d_index[1]+start_width_in_output_tensor)), int(ratio_h*max_2d_index[0])
                vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
                # Write prediction result
                f.write(f'{frame_count-(num_frame*batch_size)+i},{vis},{cx_pred},{cy_pred}\n')

            else:
                cx_pred, cy_pred = 0, 0
                vis = 0
                f.write(f'{frame_count - (num_frame * batch_size) + i},{vis},{cx_pred},{cy_pred}\n')

f.flush()
os.fsync(f.fileno())
print('推理结束，已生成csv文件')


load_csv = False
if os.path.isfile(csv_path) and csv_path.endswith('.csv'):
    load_csv = True
else:
    print("Not a valid csv file! Please modify path in parser.py --csv_path")

# acquire video info
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if load_csv:
    info = load_info(csv_path)
    if len(info) != n_frames:
        print("len(info) != n_frames",len(info) , n_frames)
        print("Number of frames in video and dictionary are not the same!")
        print("Fail to load, create new dictionary instead.")
        info = {
            idx: {
                'Frame': idx,
                'Ball': 0,
                'x': -1,
                'y': -1
            } for idx in range(n_frames)
        }
    else:
        print("Load labeled dictionary successfully.")
else:
    print("Create new dictionary")
    info = {
        idx: {
            'Frame': idx,
            'Ball': 0,
            'x': -1,
            'y': -1
        } for idx in range(n_frames)
    }


def ball_label(event, x, y, flags, param):
    global frame_no, info, image
    if event == cv2.EVENT_LBUTTONDOWN:

        info[frame_no]['x'] = x
        info[frame_no]['y'] = y
        info[frame_no]['Ball'] = 1

    elif event == cv2.EVENT_MBUTTONDOWN:
        info[frame_no]['x'] = -1
        info[frame_no]['y'] = -1
        info[frame_no]['Ball'] = 0


saved_success = False
frame_no = 0
cap = cv2.VideoCapture(video_file)
_, image = cap.read()
image = show_image(image, 0, info[0]['x'], info[0]['y'])
while True:
    leave = 'y'
    cv2.imshow('imgLabel', image)
    cv2.setMouseCallback('imgLabel', ball_label)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        if not saved_success:
            print("You forget to save file!")
            while True:
                leave = str(input("Really want to leave without saving? [Y/N]"))
                leave = leave.lower()
                if leave != 'y' and leave != 'n':
                    print("Please type 'y/Y' or 'n/N'")
                    continue
                elif leave == 'y':
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Exit label program")
                    sys.exit(1)
                elif leave == 'n':
                    break

        if leave == 'y':
            cap.release()
            cv2.destroyAllWindows()
            print("Exit label program")
            sys.exit(1)

    elif key == ord('s'):
        saved_success = save_info(info, video_path, w, h)

    elif key == ord('n'):
        if frame_no >= n_frames - 1:
            print("This is the last frame")
            continue
        frame_no += 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('p'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no -= 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('f'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no = 0
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('l'):
        if frame_no == n_frames - 1:
            print("This is the last frame")
            continue
        frame_no = n_frames - 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('>'):
        if frame_no + 36 >= n_frames - 1:
            print("Reach last frame")
            frame_no = n_frames - 1
        else:
            frame_no += 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('<'):
        if frame_no - 36 <= 0:
            print("Reach first frame")
            frame_no = 0
        else:
            frame_no -= 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))
    else:
        image = go2frame(cap, frame_no, info)

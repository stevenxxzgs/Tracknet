import os
import json
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import Badminton_Dataset
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--input_type', type=str, default='2d', choices=['2d', '3d'])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()
param_dict = vars(args)

num_frame = args.num_frame
input_type = args.input_type
batch_size = args.batch_size
debug = args.debug


# Load dataset
print(f'Data dir: {data_dir}')
print(f'Data input type: {input_type}')
train_dataset = Badminton_Dataset(root_dir=data_dir, split='train', mode=input_type, num_frame=num_frame, slideing_step=1, debug=debug)
eval_test_dataset = Badminton_Dataset(root_dir=data_dir, split='test', mode=input_type, num_frame=num_frame, slideing_step=num_frame, debug=debug)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=False) #已更改pin_memory=False
eval_loader = DataLoader(eval_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=False) #已更改pin_memory=False

data_prob = tqdm(train_loader)
epoch_loss = []
for epoch in range(0, 3):
    for step, (i, x, y, c) in enumerate(data_prob):
        print(i)
        # print(x)
        # print(y)
        print(c)

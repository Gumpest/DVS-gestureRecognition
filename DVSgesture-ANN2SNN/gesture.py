import numpy as np
import torch
import os
from torch.utils.data import Dataset
#import pandas as pd
#from torchvision import transforms, utils
#import struct
#import cv2
# dt = 30, step = 3
#自定义数据加载类
class DVSGesture(Dataset):
    def __init__(self, path, train, step, dt):
        super(DVSGesture, self).__init__()

        self.dt = dt
        self.step = step
        self.win = dt * step
        dx = 10  # 1280 // 128
        dy = 7   # 800 // 128

        self.len = 0
        if train == True:
            self.len = 5*10 #  4categories * 10files
        else:
            self.len = 5*2
        
        self.eventflow = np.zeros(shape=(self.len, self.step, 128, 128))
        self.label = np.zeros(shape=(self.len, 5))
        
        filenum = 0
        
        for num in range(5):
            dir = os.path.join(path, str(num))
            files = os.listdir(dir)
            for file in files:
                file_dir = os.path.join(dir,file)
                events = np.loadtxt(file_dir, delimiter=',')
                all_x = events[:, 1]//dx
                all_y = events[:, 0]//dy
                all_p = events[:, 3]
                all_ts = events[:, 4]
                all_ts = np.uint32(np.around(all_ts/1000))

                win_indices = np.where((all_ts<self.win)&(all_p !=0)) # select t in win
                win_indices = win_indices[0]

                for i in range(len(win_indices)):
                    index = int(win_indices[i])
                    self.eventflow[filenum, int(all_ts[index]//dt), int(all_x[index]), int(all_y[index])] = 1  # 1 for an event, 0 for nothing
                self.label[filenum] = np.eye(5)[num]  # one-hot label
                filenum += 1
            print("Done file:" + str(num))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eventflow[idx, :, :, :]
        y = self.label[idx]

        return (x, y)


# Return Data that model can read
def celexEvalOnce(path, step, dt):
    win = step * dt
    dx = 10 # 1280 // 128
    dy = 7 # 800 // 128
    file = np.loadtxt(path, delimiter=',')
    x = file[:, 1]//dx
    y = file[:, 0]//dy
    polar = file[:, 3]
    t = file[:, 4] // 1000

    #eventflow = np.zeros(shape=(1, 34, 34, step))  # alternative shape
    eventflow = np.zeros(shape=(1, step, 128, 128))  # alternative shape
    win_indices = np.where((t < win) & (polar != 0)) 
    win_indices = win_indices[0] # squeeze tuple
    for i in range(len(win_indices)): 
        index = int(win_indices[i])
        #threshold = 2.5
        eventflow[0, int(t[index] // dt), int(x[index]), int(y[index])] = 1  # 1 for an event, 0 for nothing
        #if eventflow[0, int(max(0, polar[index])), int(x[index] // dx), int(y[index] // dy), int(t[index] // dt)] < threshold:
            #eventflow[0, int(max(0, polar[index])), int(x[index] // dx), int(y[index] // dy), int(t[index] // dt)] += 1  # 1 for an event, 0 for nothing


    #print(np.mean(eventflow))
    #print(np.median(eventflow))
    #eventflow = eventflow // threshold
    #print(np.mean(eventflow))
    #print(np.max(eventflow))

    eventflow = torch.from_numpy(eventflow)

    return eventflow

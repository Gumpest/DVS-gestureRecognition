import numpy as np
import torch
import os
from torch.utils.data import Dataset
#import pandas as pd
#from torchvision import transforms, utils
#import struct
#import cv2

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
        
        self.eventflow = np.zeros(shape=(self.len, 2, 128, 128, self.step))
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

                win_indices = np.where((all_ts<self.win)&(all_p !=0))  # select t in win
                win_indices = win_indices[0]

                for i in range(len(win_indices)):
                    index = int(win_indices[i])
                    self.eventflow[filenum, int(max(0, all_p[index])), int(all_x[index]), int(all_y[index]), int(all_ts[index]//dt)] = 1  # 1 for an event, 0 for nothing
                self.label[filenum] = np.eye(5)[num]  # one-hot label
                filenum += 1
            print("Done file:" + str(num))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eventflow[idx, :, :, :, :]
        y = self.label[idx]

        return (x, y)


# Return Data that model can read
def celexEvalOnce(path, step, dt, genVideo=False):
    win = step * dt
    dx = 10 # 1280 // 128
    dy = 7 # 800 // 128
    file = np.loadtxt(path, delimiter=',')
    x = file[:, 1]
    y = file[:, 0]
    polar = file[:, 3]
    t = file[:, 4] // 1000

    #eventflow = np.zeros(shape=(1, 34, 34, step))  # alternative shape
    eventflow = np.zeros(shape=(1, 2, 128, 128, step))  # alternative shape
    win_indices = np.where((t < win) & (polar != 0)) 
    win_indices = win_indices[0] # squeeze tuple
    for i in range(len(win_indices)): 
        index = int(win_indices[i])
        #threshold = 2.5
        eventflow[0, int(max(0, polar[index])), int(x[index] // dx), int(y[index] // dy), int(t[index] // dt)] = 1  # 1 for an event, 0 for nothing
        #if eventflow[0, int(max(0, polar[index])), int(x[index] // dx), int(y[index] // dy), int(t[index] // dt)] < threshold:
            #eventflow[0, int(max(0, polar[index])), int(x[index] // dx), int(y[index] // dy), int(t[index] // dt)] += 1  # 1 for an event, 0 for nothing


    #print(np.mean(eventflow))
    #print(np.median(eventflow))
    #eventflow = eventflow // threshold
    #print(np.mean(eventflow))
    #print(np.max(eventflow))

    eventflow = torch.from_numpy(eventflow)

    if genVideo:
        fps = 30
        size = (512, 512)

        #fourcc = cv2.CV_FOURCC('M', 'J', 'P', 'G')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        #fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
        videowriter = cv2.VideoWriter("eval.avi", fourcc, fps, size)
        
        
        for i in range(step): # or win
            neg_raw = eventflow[0, 0, :, :, i].numpy()
            pos_raw = eventflow[0, 1, :, :, i].numpy()
            neg_img = np.zeros((128, 128, 3), dtype='uint8')
            pos_img = np.zeros((128, 128, 3), dtype='uint8')
            indices = np.where(neg_raw == 1)
            neg_img[indices] = [255, 255, 255]
            #cv2.imwrite('./image/negimg_'+str(i)+'.jpg', neg_img)
            indices = np.where(pos_raw == 1)
            pos_img[indices] = [255, 255, 255]
            #cv2.imwrite('./image/posimg_'+str(i)+'.jpg', pos_img)
            img = neg_img + pos_img

            img.dtype = 'uint8'
            #print(img.shape)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
            img = cv2.transpose(img)
            #print(img.shape)
            cv2.imwrite('./image/img_'+str(i)+'.jpg', img)

            videowriter.write(img)
        videowriter.release()

    return eventflow

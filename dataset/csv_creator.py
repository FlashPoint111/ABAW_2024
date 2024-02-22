import cv2
import os
import numpy as np
import csv
import pandas as pd

def video_split(train_set, test_set, video_path, output_path, subset, label_path):
    video_path = video_path+'.mp4' if os.path.exists(video_path+'.mp4') else video_path+'.avi'
    videoinfo = cv2.VideoCapture(video_path)
    if subset == 'Train_Set':
        train_set.loc[len(train_set)] = [video_path, label_path, videoinfo.get(cv2.CAP_PROP_FPS)]
    else:
        test_set.loc[len(test_set)] = [video_path, label_path, videoinfo.get(cv2.CAP_PROP_FPS)]


video_path = "../batch1"
label_path = "../EXPR_Classification_Challenge"
output_path = "../"
train_set = pd.DataFrame(data=None,columns=['video_path', 'label_path', 'fps'])
test_set = pd.DataFrame(data=None,columns=['video_path', 'label_path', 'fps'])
for root, dirs, files in os.walk(label_path):
    for name in files:
        if 'left' in name or 'right' in name:
            video_split(train_set, test_set, os.path.join(video_path,name.split('_')[0]), output_path, root.split('\\')[-1], os.path.join(root, name))
        else:
            video_split(train_set, test_set, os.path.join(video_path,name.split('.')[0]), output_path, root.split('\\')[-1], os.path.join(root, name))

train_set.to_csv(r'train.csv', index=False, encoding='utf-8')
test_set.to_csv(r'test.csv', index=False, encoding='utf-8')
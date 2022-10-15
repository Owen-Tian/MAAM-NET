import random
#import cv2
import os
import numpy as np

from joint_trans import *

#this function is to make image-flow pairs
def make_dataset(root_ori,root_flo,train=True,if_sub=True):
    if root_ori[-1]!='/':
        root_ori+='/'
    if root_flo[-1]!='/':
        root_flo+='/'
    dataset=[]
    if train:
        root_ori+='training/'
        root_flo+='training/'
    else:
        root_ori+='testing/'
        root_flo+='testing/'
    if if_sub:
        #root contains different sub dirs       
        sub_dirs_length=len(sorted(os.listdir(root_ori)))
        sub_dirs_ori=sorted(os.listdir(root_ori))
        sub_dirs_flo=sorted(os.listdir(root_flo))
        for sub in range(sub_dirs_length):

            sub_images_length=len(sorted(os.listdir(root_flo+sub_dirs_flo[sub])))
            sub_images_ori=sorted(os.listdir(root_ori+sub_dirs_ori[sub]))
            sub_images_flo=sorted(os.listdir(root_flo+sub_dirs_flo[sub]))

            for image in range(sub_images_length):
                ori=root_ori+sub_dirs_ori[sub]+'/'+sub_images_ori[image]
                flo=root_flo+sub_dirs_flo[sub]+'/'+sub_images_flo[image]
                dataset.append([ori,flo])
    else:
        #root not contains different sub dirs 
        sub_images_length=len(sorted(os.listdir(root_flo)))
        sub_images_ori=sorted(os.listdir(root_ori))
        sub_images_flo=sorted(os.listdir(root_flo))
        
        for image in range(sub_images_length):
            ori=root_ori+sub_images_ori[image]
            flo=root_flo+sub_images_flo[image]
            dataset.append([ori,flo])

    return dataset

def preprocess(image_data_batch,dataset=None,h=180,w=320,c=1,gray=False,aug=False):
    te_raw=image_data_batch
    for_feed_images=[]
    for_feed_flows=[]
    for raw in te_raw: 
        
        raw_image,raw_flow=open_image(raw[0],raw[1],h=h,w=w,c=c,gray=gray)
               
        if aug:
            raw_image,raw_flow=image_joint_flip(raw_image,raw_flow,tag='v')
            raw_image,raw_flow=image_joint_flip(raw_image,raw_flow,tag='h')
            raw_image,raw_flow= image_joint_noise(raw_image,raw_flow)         
            raw_image,raw_flow= image_joint_randcrop(raw_image,raw_flow,size=[h+20,w+30])         
            raw_image,raw_flow= image_joint_rotate(raw_image,raw_flow)
              
        for_feed_images.append(np.float32(raw_image).reshape(h,w,-1))           
        for_feed_flows.append(raw_flow)
        
    for_feed_images=np.float32(for_feed_images)
    for_feed_flows=np.float32(for_feed_flows)

    return for_feed_images,for_feed_flows


class DataLoader(object):
    def __init__(self,root_ori,root_flo,train=True,batch_size=1,if_sub=True):
        self.imgs_data=make_dataset(root_ori,root_flo,train,if_sub=if_sub)
        self.batch_size=batch_size
        self.index=0
        self.max_index=len(self.imgs_data)//self.batch_size
        
    def __getitem__(self,index):
        return self.imgs_data[index]

    def all(self):
        return self.imgs_data
    
    #Before every training epoch,a shuffle is a must
    def shuffle(self):
        random.shuffle(self.imgs_data)
        self.index=0
    
    def next_batch(self):
        t=[]
        if self.index<self.max_index:
            t=self.imgs_data[self.index*self.batch_size:self.index*self.batch_size+self.batch_size]
            self.index+=1
        return t



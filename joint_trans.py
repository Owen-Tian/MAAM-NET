import random
import cv2
import os
from PIL import Image, ImageOps
from skimage import util
import numpy as np
from flow_utils import *

def image_joint_rotate(image1,image2,image3=[],degree=0): 
    if random.random()>0.5:
        #print('rotate')
        if image3!=[]:
            return cv2.flip(image1,-1),cv2.flip(image2,-1),cv2.flip(image3,-1)
            #image1.transpose(Image.ROTATE_180),image2.transpose(Image.ROTATE_180),image3.transpose(Image.ROTATE_180)
        return cv2.flip(image1,-1),cv2.flip(image2,-1)
            #image1.transpose(Image.ROTATE_180),image2.transpose(Image.ROTATE_180)
    else:
        if image3!=[]:
            return image1,image2,image3
        return image1,image2


def image_joint_flip(image1,image2,image3=[],tag='v'):
    #tag={'v','h'} 
    if tag=='v':
        
        if random.random()>0.5:
            #print('flip v')
            if image3!=[]:
                return cv2.flip(image1,0),cv2.flip(image2,0),cv2.flip(image3,0)
                #ImageOps.flip(image1),ImageOps.flip(image2),ImageOps.flip(image3)
                #image1.rotate(90,Image.BILINEAR),image2.rotate(90,Image.NEAREST),image3.rotate(90,Image.NEAREST)
            return cv2.flip(image1,0),cv2.flip(image2,0)
                #ImageOps.flip(image1),ImageOps.flip(image2)
                #image1.rotate(90,Image.BILINEAR),image2.rotate(90,Image.NEAREST)
        else:
            if image3!=[]:
                return image1,image2,image3
            return image1,image2
    elif tag=='h':
        
        if random.random()>0.5:
            #print('flip h')
            if image3!=[]:
                return cv2.flip(image1,1),cv2.flip(image2,1),cv2.flip(image3,1)
                #ImageOps.mirror(image1),ImageOps.mirror(image2),ImageOps.mirror(image3)
                #image1.rotate(180,Image.BILINEAR),image2.rotate(180,Image.NEAREST),image3.rotate(180,Image.NEAREST)
            return cv2.flip(image1,1),cv2.flip(image2,1)
                #ImageOps.mirror(image1),ImageOps.mirror(image2)
                #image1.rotate(180,Image.BILINEAR),image2.rotate(180,Image.NEAREST)
        else:
            if image3!=[]:
                return image1,image2,image3
            return image1,image2
    
def gasuss_noise(image, mean=0, var=0.02):
    
    image = np.array(image, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + noise
    
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    
    return out

def image_joint_noise(image1,image2,image3=[],noise=0.02):
    if random.random()>0.5:
        #print('noise')
        seeds=random.randint(0,100000)
        np.random.seed(seed=seeds)
        noise_img1=gasuss_noise(image1)
        if image3!=[]:
            
            return noise_img1,image2,image3
        return noise_img1,image2
    else:
        if image3!=[]:
            
            return image1,image2,image3
        return image1,image2

def image_joint_randcrop(image1,image2,image3=[],size=[260,390]): 
    #image1 is original image,2 and 3 is flow image  
    if random.random()>0.5:
        #print('crop')
        w,h=image1.shape[:2] #original size
        tw,th=size #resize shape
        
        image1=cv2.resize(image1,(th,tw))
        image2=cv2.resize(image2,(th,tw))

        x1=random.randint(0,tw-w)
        y1=random.randint(0,th-h)
        if image3!=[]:
            return image1[x1:x1+w,y1:y1+h,:],image2[x1:x1+w,y1:y1+h,:],image3[y1:y1+h,x1:x1+w,:]
        if len(image1.shape)>3:
            return image1[x1:x1+w,y1:y1+h,:],image2[x1:x1+w,y1:y1+h,:]
        return image1[x1:x1+w,y1:y1+h],image2[x1:x1+w,y1:y1+h,:]
    else:
        if image3!=[]:
            return image1,image2,image3
        return image1,image2

def open_image(image1,image2,h=180,w=320,c=1,gray=False):   
    if not gray:
        image1=cv2.imread(image1)
    else:
        image1=cv2.imread(image1,flags=cv2.IMREAD_GRAYSCALE)

    if (image1.shape[0],image1.shape[1])!=(h,w):
        image1=cv2.resize(image1,(w,h))
   
    image1=image1/127.5-1

    if image2.split('.')[-1]=='flo':
        image2=readFlow(image2)
    elif image2.split('.')[-1]=='npy':
        image2=np.load(image2)
    return image1,image2


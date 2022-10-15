import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

def func(data_source,target='../3',dataset=0):
    images_name=[]
    flows_name=[]
    for i in sorted(os.listdir(data_source)):
        
        back=i.split('.')[-1]
        if back=='npy':
            if i.find('r')==0:
                flows_name.append(data_source+'/'+i)
            else:
                images_name.append(data_source+'/'+i)
    print(images_name[1],flows_name[2])
    for i in range(len(flows_name)):
        data1=images_name[i]
        data2=flows_name[i]
        print(data1,data2)
        my_colormap = LinearSegmentedColormap.from_list("", ["white", "red"])
        buff1=np.load(data1)#image
        buff2=np.load(data2) #flow
        buff1=np.mean(buff1,axis=2)
        if dataset==0:
            s=(240,360)
        else:
            s=(180,320)
        assert buff1.shape==s
        buff2=np.mean(buff2[:,:,:2],axis=2)
        assert buff2.shape==s
        """
        buff11=(buff1-np.min(buff1))/(np.max(buff1)-np.min(buff1))
        sns.heatmap(buff11,cmap=my_colormap)
        plt.axis('off')
        plt.savefig(target+'/'+str(i)+'_image.png',bbox_inches='tight',pad_inches=0)
        plt.clf()
        
        buff22=(buff2-np.min(buff2))/(np.max(buff2)-np.min(buff2))
        sns.heatmap(buff22,cmap=my_colormap)
        plt.axis('off')
        plt.savefig(target+'/'+str(i)+'_flow.png',bbox_inches='tight',pad_inches=0)
        plt.clf()
        """
        if dataset==0:
            mulit=10
        elif dataset==1:
            mulit=100
        else:
            mulit=3
        buff3=mulit*buff1+buff2
        buff3=(buff3-np.min(buff3))/(np.max(buff3)-np.min(buff3))
        buff3-=0.7
        if dataset==0:
            w,h=240,360
        else:
            w,h=180,320
        for k in range(w):
            for j in range(h):
                if buff3[k][j]<=0:
                    buff3[k][j]=0
                else:
                    buff3[k][j]=1 
        sns.heatmap(buff3,cmap=my_colormap)
        plt.axis('off')
        plt.savefig(target+'/'+str(i)+'_total.png',bbox_inches='tight',pad_inches=0)
        plt.clf()

func('./New folder1',target='../3',dataset=0)



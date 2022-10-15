import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from kh_tools import read_lst_images

def get_locations(source_image_name='001.jpg',dataset='ped2'):
    if dataset=='ped2':
        patch_size=[30,20]
        stride=[8,5]
    elif dataset=='Avenue':
        patch_size=[85,40]
        stride=[5,5]
    elif dataset=='shanghaiTech':
        patch_size=[20,28]
        stride=[5,8]
    res_name='locs_'+dataset
    print(patch_size,stride)
    _,locs=read_lst_images([source_image_name],patch_size,stride)
    print(len(locs))
    np.save(res_name,locs)
    print('save locs successful!')



def load_detection_file(dir='./',locations='./locs_ped2.npy',threshold=0.5,dataset='ped2'):
    results=np.load(dir)
    print('original shape:',results.shape)
    if dataset=='Avenue':
        results=results.reshape(results.shape[0]*results.shape[1],results.shape[2]*results.shape[3])
    print('adjusted shape:',results.shape)

    samples,los=results.shape[0],results.shape[1]
    
    #adjust score
    if dataset=='Avenue':
        error=results-np.min(results)
        score=1.-error/np.max(error)
    elif dataset=='ped2':
        score=results

    
    #obtain locations   
    locations=np.load(locations)
    locs=[]
    for i in range(samples):
        ls=[]
        for j in range(los):
            if score[i][j]<threshold:
                ls.append([locations[j][0],locations[j][1]])
        
        locs.append(ls)

    #print(len(locs))
    #np.save('locs_details_'+dataset,np.array(locs,dtype=object))
    print('deal locs_details successful!')
    return locs


def read_file(file_name,tag=1):
    if tag==1:
        return cv2.imread(file_name)
    else:
        return cv2.imread(file_name,flags=cv2.IMREAD_GRAYSCALE)
def load_files(image_source='./',gt_source='./',dataset='ped2'):
    files=[]
    files_gt=[]
    for file_dirs in sorted(os.listdir(image_source)):
        for file in sorted(os.listdir(image_source+'/'+file_dirs)):
            files.append(image_source+'/'+file_dirs+'/'+file)
        if dataset=='ped2':
            files=files[:-1]
        else:
            files=files[:-2]

    if dataset=='ped2':
        for file_dirs in sorted(os.listdir(gt_source)):
            for file in sorted(os.listdir(gt_source+'/'+file_dirs)):
                files_gt.append(gt_source+'/'+file_dirs+'/'+file)
            files_gt=files_gt[:-1]
        np.save('file_gt_names_'+dataset,files_gt)

    print(len(files))
    np.save('file_names_'+dataset,files)
    
    print('save file names successfule!')

def plot_images(locs_details='./',file_names='./',gt_names='./',dataset='ped2',save_dirs='./'):
    if dataset=='ped2':
        patch_size=[30,20]
        stride=[8,5]
    elif dataset=='Avenue':
        patch_size=[85,40]
        stride=[5,5]
    elif dataset=='shanghaiTech':
        patch_size=[20,28]
        stride=[5,8]
    file_names=np.load(file_names)
    gt_names=np.load(gt_names)
    ious=[]
    if dataset=='Avenue':
        locs_details=np.load(locs_details,allow_pickle=True)
    
    for i in range(len(file_names)):
        print(len(file_names),file_names[i])
        if locs_details[i]==[]:
            continue
        source_image=read_file(file_names[i])
        
        if dataset=='ped2':
            gt_image=read_file(gt_names[i],tag=2)

            gt_image=np.float32(gt_image.reshape(1,-1)[0])

            h,w=source_image.shape[:2]
            mask=np.zeros(source_image.shape[:2])     
            for dot in locs_details[i]:
                if dot !=[]:
                    mask[dot[0]-stride[0]:dot[0]-stride[0]+patch_size[0],dot[1]-stride[1]:dot[1]-stride[1]+patch_size[1]]=255
                
            source_image=np.float32(source_image.reshape(-1,3))
            mask=np.float32(mask.reshape(1,-1)[0])
            for j in range(len(mask)):
                if mask[j]==255:
                    source_image[j]+=[0,0,255]
                if gt_image[j]!=0:
                    source_image[j]+=[0,255,0]
        elif dataset=='Avenue':
            
            
            source_image=cv2.resize(source_image,(320,180))
            gt_image=gt_names[i].reshape(1,-1)[0]
            print(gt_image.shape,'111111')
            h,w=source_image.shape[:2]
            mask=np.zeros(source_image.shape[:2])     
            for dot in locs_details[i]:
                if dot !=[]:
                    mask[dot[0]-stride[0]:dot[0]-stride[0]+patch_size[0],dot[1]-stride[1]:dot[1]-stride[1]+patch_size[1]]=255
                
            source_image=np.float32(source_image.reshape(-1,3))
            mask=np.float32(mask.reshape(1,-1)[0])
            for j in range(len(mask)):
                if mask[j]==255:
                    source_image[j]+=[0,0,255]
                if gt_image[j]!=0:
                    source_image[j]+=[0,255,0]
        print(np.sum(mask),np.sum(gt_image))
        if np.sum(mask)!=0 or np.sum(gt_image)!=0:
            cv2.imwrite(save_dirs+'/'+str(i)+'_'+file_names[i].split('/')[-1].split('.')[0]+'.png',source_image.reshape(h,w,3))
        
            a1=np.sum(np.logical_and(mask,gt_image))
            a2=np.sum(np.logical_or(mask,gt_image))
            
            print(a1,a2,a1/(a2+1e-14))
            ious.append(a1/(a2+1e-14))
       
    #plt.plot([i for i in range(1998)],ious)
    print(ious)
    #plt.plot([i for i in range(20)],ious)
    #plt.show()
    count=0
    for i in ious:
        if i!=0:
            count+=1
    print(count,np.sum(ious)/(count+1e-14))
    np.save('ious.npy',ious)

#load_files('/home/junwen/zfs/dataset/UCSDped2/testing','/home/junwen/zfs/dataset/UCSDped2/testing_map',dataset='ped2')
#load_files('/home/junwen/zfs/dataset/Avenue/testing',dataset='Avenue')


#get_locations('/home/junwen/zfs/dataset/UCSDPed2/testing/Test001/001.tif')
#get_locations('./1.jpg',dataset='Avenue')

#a=load_detection_file('result_spnor.npy',locations='locs_ped2.npy',threshold=0.5,dataset='ped2')
a=load_detection_file('result_avenue.npy',locations='locs_Avenue.npy',threshold=0.8,dataset='Avenue')
np.save('locs_details_avenue.npy',np.array(a,dtype=object))
#plot_images(locs_details=a,file_names='file_names_ped2.npy',gt_names='file_gt_names_ped2.npy',dataset='ped2',save_dirs='/home/junwen/zfs/experiment/ped2_show/')
plot_images(locs_details='locs_details_avenue.npy',file_names='file_names_Avenue.npy',gt_names='./gt_avenue.npy',dataset='Avenue',save_dirs='/home/junwen/zfs/experiment/Avenue_show/')

import numpy as np
from scipy import io
import os
import cv2

def splitnpy2totalnpy(dir='./'):
    res=[]
    for file in sorted(os.listdir(dir)):
        temp=np.load(os.path.join(dir,file))
        print(temp.shape)
        for image in temp:
            image=cv2.resize(image,(320,180))
            res.append(image)
        res=res[:-1]
    print(np.array(res).shape)
    np.save('gt_shanghaiTech.npy',res)


#splitnpy2totalnpy(r'C:\Users\IAIR\Desktop\fight_4_pr_joural\auxiliary material\4paper\dataset\ShanghaiTechDataset\Testing\ano_pred_cvpr2018\ShanghaiTechDataset\Testing\test_pixel_mask')  
def mats2npys(dir='./'):
    res_npys=[]
    for file in range(1,22):
        print(dir+'/'+str(file)+'_label.mat')
        mat = io.loadmat(dir+'/'+str(file)+'_label.mat')
        
        #print(mat.keys())
        # 可以用values方法查看各个cell的信息
        #print(mat.values())

        # 可以用shape查看维度信息
        #print(mat['volLabel'].shape)
        # 注意，这里看到的shape信息与你在matlab打开的不同
        # 这里的矩阵是matlab打开时矩阵的转置
        # 所以，我们需要将它转置回来
        mat_t = np.transpose(mat['volLabel'])
        # mat_t 是numpy.ndarray格式
        print('1 shape:',mat_t.shape)
        mat_t=mat_t.reshape(1,-1)[0]
        print('1 shape:',mat_t.shape)
        for mat_file in mat_t:
            temp=cv2.resize(mat_file,(320,180))
            print(temp.shape,'tempshape')
            res_npys.append(temp)
        print(np.array(res_npys).shape)
        res_npys=res_npys[:-2]
        print(np.array(res_npys).shape)
        # 再将其存为npy格式文件
        

    print(np.array(res_npys).shape)
    np.save('gt_avenue1.npy', np.array(res_npys))
    #np.save('gt_avenue',np.array(res_npys))

mats2npys(r'C:\Users\IAIR\Desktop\fight_4_pr_joural\auxiliary material\4paper\dataset\Avenue Dataset\ground_truth_demo\testing_label_mask')
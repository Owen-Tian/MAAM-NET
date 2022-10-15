import matplotlib.pyplot as plt
import numpy as np
import os

def get_files(gt_source,pe_file,db=0):
    gt_file=np.load(gt_source)
    pe_file=np.load(pe_file)

    c=1 #substract one to insure the sample consistency
    if db==0:
        dirs=[180, 180, 150, 180, 150, 180, 180, 180, 120, 150, 180, 180]
    elif db==1:
        dirs=[1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841, 472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,  76]
        c=2
    else:
        dirs=np.load('../dirs.npy')       
    hty=0
    num=1
    for i in dirs:
        i-=c
        gt=gt_file[hty:hty+i]
        pe=pe_file[hty:hty+i]
        print(hty,hty+i)
        plots(gt,pe,i,db,num)
        hty+=i
        num+=1
            
def plots(gt,pe,i,db,num):
    plt.axis([0,i,0,1])
    plt.xticks(np.arange(1,i,1))
    plt.yticks(np.arange(0,1,0.2))
    plt.xticks([])
    #plt.yticks([])
    
    res=find_range(pe)
    for index in res:
        plt.axvspan(index[0], index[1], facecolor='#99CCFF', alpha=0.5)
    
    
    plt.plot([i for i in range(i)],gt)
    
    plt.savefig(str(db)+'_'+str(num)+'.jpg')
    plt.clf()
    #plt.show()

def get_dirs(source):
    #001,002,003...
    d=[]
    for dir in sorted(os.listdir(source)):
        d.append(len(os.listdir(os.path.join(source,dir))))
    
    np.save('dirs.npy',d)
    return d

#print(get_dirs('/home/junwen/zfs/dataset/shanghaiTech/testing/'))

def find_range(gt):
    pe=gt
    pe=list(pe)
    te=pe  
    res=[]
    while te.count(1)!=0:
        h=len(pe)-len(te)
        #print('h:',h)
        st=te.index(1)
        if te[st:].count(0)!=0:
            ct=te[st:].index(0)
        else:
            ct=len(te)-st
        #print('h:',h,'st:', st, 'ct:' ,ct)      
        res.append([h+st,h+st+ct])
        te=pe[h+st+ct:]
    return res

get_files('../pd.npy','../ped.npy',0)
get_files('../avenue.npy','../ave.npy',1)
get_files('../sht1.npy','../sht.npy',2)
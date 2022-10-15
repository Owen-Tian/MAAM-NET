import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import MultipleLocator

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
        dirs=np.load('dirs.npy')       
    hty=0
    num=1
    sx=1
    for i in dirs:

        i-=c
        gt=gt_file[hty:hty+i]
        pe=pe_file[hty:hty+i]
        print(hty,hty+i)
        if db==0:
            if sx==2 or sx==4:
                plots(gt,pe,i,db,num)
        elif db==1:
            if sx==5 or sx==6:
                if sx==5:
                    plots(gt,pe,i,db,num,0)
                else:
                    plots(gt,pe,i,db,num,0)
        else:
            if sx==25 or sx==35:# or sx==65:
                plots(gt,pe,i,db,num,0)
        hty+=i
        num+=1
        sx+=1
            
def plots(gt,pe,i,db,num,min_index=0):
    plt.axis([0,i,0,1-min_index])
    plt.xticks(np.arange(1,i,1))
    plt.yticks(np.arange(0,1-min_index,0.2))
    plt.xticks([])
    plt.yticks([])
    
    res=find_range(pe)
    for index in res:
        plt.axvspan(index[0], index[1], facecolor='#7bbfea')
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=0.85,bottom=0.65,left=0.01,right=0.99,hspace=0.3,wspace=0.6) 
    plt.plot([i for i in range(i)],1-gt)
    
    #plt.savefig('./avenue/'+str(db)+'_'+str(num)+'.jpg')
    #plt.clf()
    plt.show()

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

get_files('./pd.npy','./ped.npy',0) #0
get_files('./avenue.npy','./ave.npy',1) #0.2 0.4
get_files('./sht1.npy','./sht.npy',2)


def plot_line_charts_w_two_axis(y_sub1,y_sub2,label,remark):
    ###
        # y_sub1: main y axis data
        # y_sub2: vice y axis data
        # label: x axis label
        # remark: x axis remark
    ###
    data1=y_sub1
    data2=y_sub2
    axis1=[1,2,3,4,5]

    fig,ax1=plt.subplots()
    ax2=ax1.twinx()

    ax1.plot(axis1,data1,'#F39C12',linewidth=3,marker='^',markersize=10)
    ax2.plot(axis1,data2,'skyblue',linewidth=3,marker='s',markersize=10)
    ax1.set_xticks(axis1)

    y_ticks = np.arange(0.970, 0.980, 0.002)
    
    if remark=='ζ':    
        y2_ticks= np.arange(0.890, 0.910, 0.004)
    else:
        y2_ticks= np.arange(0.865, 0.915, 0.01)
    ax1.set_yticks(y_ticks)
    ax2.set_yticks(y2_ticks)
    
    ax1.set_xlabel(remark,{'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        })
    ax1.set_ylabel('AUC',{'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 15,
        })

    ax1.set_xticklabels(label)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    
    labels = ax1.get_xticklabels() + ax1.get_yticklabels() +ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax1.legend(['  Ped2  '],loc=(0.03,0.88),prop = {'size':13,'family' : 'Times New Roman'})
    ax2.legend(['Avenue'],loc=(0.03,0.78),prop = {'size':13,'family' : 'Times New Roman'})
    plt.show()


#y_sub1=[0.972,0.974,0.977,0.973,0.975]
#y_sub2=[0.902,0.893,0.909,0.892,0.901]
#label=[0.025,0.05,0.1,0.2,0.4]
#plot_line_charts_w_two_axis(y_sub1,y_sub2,label,'ζ')

#y_sub1=[0.973,0.975,0.977,0.972,0.973]
#y_sub2=[0.871,0.902,0.909,0.894,0.902]
#label=[0.00025,0.0005,0.001,0.002,0.004]
#plot_line_charts_w_two_axis(y_sub1,y_sub2,label,'γ')



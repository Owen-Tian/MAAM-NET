# coding: utf-8
import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn import metrics

from memory_module import memory
from Dataset import DataLoader,preprocess
from model import encoder,decoder,decoder_wo_mem
from flow_utils import visulize_flow_file

def inference(gt_file,scores,pos_label):

    fpr,tpr,thresholds=metrics.roc_curve(gt_file,np.array(scores).reshape(1,-1)[0],pos_label=pos_label)
    
    auc=metrics.auc(fpr,tpr)
    eer=brentq(lambda x: 1.-x - interp1d(fpr,tpr)(x),0. ,1. )

    return auc, eer


def cos_sim(input1,input2):

    numerator=tf.matmul(input1,input2,transpose_b=True)
    
    denominator=tf.matmul(input1**2,input2**2,transpose_b=True)
    w=(numerator+1e-12)/(denominator+1e-12)
    return w


def test(testing_raw, start_model_idx=0, mem_capacity=2000, batch_size=9,dataset='ped',gray=False,train=False):
    #hyper parameters  
    if gray:
        c=1
    else:
        c=3

    if dataset=='ped2':
        h,w,c,patch_h,patch_w,patch_stride_h,patch_stride_w,hp1,hp2,test_samples=240,360,c,30,20,5,8,1,10,1998  
    elif dataset=='avenue':
        h,w,c,patch_h,patch_w,patch_stride_h,patch_stride_w,hp1,hp2,test_samples=180,320,c,85,40,5,5,1,100,15282   
    elif dataset=='shanghaiTech':
        h,w,c,patch_h,patch_w,patch_stride_h,patch_stride_w,hp1,hp2,test_samples=180,320,c,20,28,5,8,10,1,40684
   
    #the one is for images,the other for flows  
    inputs_images=tf.placeholder(tf.float32,[batch_size,h,w,c])
    inputs_flows=tf.placeholder(tf.float32,[batch_size,h,w,2])
    
    #z is latent vector as a query to obtain latent vector for recons from memory
    z=encoder(inputs_images,is_training=train)

    #initialize memory
    shrink_thres=1./mem_capacity
    memo=memory(mem_capacity,z.shape[-1],shrink_thres)

    #latent is for recons for images and flows,sims is to count items numbers,sis is raw sims
    latent,sims,sis=memo.query(z)

    #obtain recons images and flows
   
    recons_images,recons_flows=decoder(latent,h=h,w=w,c=c,is_training=train)
    ##recons_images_wo_mem,recons_flows_wo_mem=decoder_wo_mem(z,h=h,w=w,c=c,is_training=train)
    
    #if need discriminator
    #dis_true,dis_true_logits=Discriminator(inputs_images,inputs_flows)
    #dis_false,dis_false_logits=Discriminator(inputs_images,recons_flows,reuse=True)

    #l2 loss of RGBs and l1 loss of Flows 
    loss_recons_image=tf.reduce_mean(tf.square(recons_images-inputs_images))
    loss_recons_flow=tf.reduce_mean(tf.abs(recons_flows-inputs_flows))

    ##loss_recons_image_wo_mem=tf.reduce_mean(tf.square(recons_images_wo_mem-inputs_images))
    ##loss_recons_flow_wo_mem=tf.reduce_mean(tf.abs(recons_flows_wo_mem-inputs_flows))
    
    #find the RGB loss in the batch
    loss_recons_image_batch=tf.reduce_mean(tf.square(recons_images-inputs_images),axis=(1,2,3))
    ##loss_recons_image_batch_wo_mem=tf.reduce_mean(tf.square(recons_images_wo_mem-inputs_images),axis=(1,2,3))

    #loss_grad
    #dy1, dx1 = tf.image.image_gradients(recons_images)
    #dy0, dx0 = tf.image.image_gradients(inputs_images)
    #loss_grad = tf.reduce_mean(tf.abs(tf.abs(dy1)-tf.abs(dy0)) + tf.abs(tf.abs(dx1)-tf.abs(dx0)))
    
    #Latent Loss
    #loss_latent=tf.maximum(tf.abs(tf.reduce_mean(latent-z))-0.1,0.0)
    loss_latent=tf.maximum(tf.abs(tf.reduce_mean(cos_sim(latent,z)))-0.1,0.0)
		
	#Sparsity Loss
    loss_sparsity=tf.reduce_mean((0-sims)*tf.log(sims+1e-12))

    #recons error, and make it into patches,wei is the filter of all-ones 
    loss_recons_images_4d=tf.square(recons_images-inputs_images)
    loss_recons_flows_4d=tf.abs(recons_flows-inputs_flows) 

    ##loss_recons_images_4d_wo_mem=tf.square(recons_images_wo_mem-inputs_images)
    ##loss_recons_flows_4d_wo_mem=tf.abs(recons_flows_wo_mem-inputs_flows) 

    #----------------------------------------------------------------
    #detection procedure

    wei = tf.ones(shape=[patch_h, patch_w, c, 1],name='split_score')
    wei_f = tf.ones(shape=[patch_h, patch_w, 2, 1],name='split_score2')       
    stride=[1, patch_stride_w, patch_stride_h, 1]    

    #recons image error
    result = tf.nn.conv2d(loss_recons_images_4d, wei , stride, 'SAME') 
    ##result_wo_mem= tf.nn.conv2d(loss_recons_images_4d_wo_mem, wei , stride, 'SAME') 
    #recons flow error         
    result_f=tf.nn.conv2d(loss_recons_flows_4d, wei_f, stride,'SAME')
    ##result_f_wo_mem=tf.nn.conv2d(loss_recons_flows_4d_wo_mem, wei_f , stride, 'SAME') 

    #single branch
    result_image_only=tf.reduce_max(result,axis=(1,2,3))
    ##result_image_only_no_mem=tf.reduce_max(result_wo_mem,axis=(1,2,3))

    result_flow_only=tf.reduce_max(result_f,axis=(1,2,3))
    ##result_flow_only_no_mem=tf.reduce_max(result_f_wo_mem,axis=(1,2,3))

    #fuse   
    result_fusion=tf.reduce_max((hp1*result_f+hp2*result)/(hp1+hp2),axis=(1,2,3))
    ##result_fusion_no_mem=tf.reduce_max((hp1*result_f_wo_mem + hp2*result_wo_mem)/(hp1+hp2),axis=(1,2,3))
    #----------------------------------------------------------------

    #total loss
    loss= loss_recons_image + 2*loss_recons_flow + 0.0003*loss_sparsity + 0.001*loss_latent
    ##loss_wo_mem= loss_recons_image_wo_mem + 2*loss_recons_flow_wo_mem
    
    #learning_rates
    #learning_rates=tf.train.exponential_decay(learning_rate=0.0001,global_step=1000,decay_steps=100,decay_rate=1./10,staircase=True,name='ls_dc')
    #learning_rates=lr

    #saves the batch moving mean and var
    update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    #with tf.control_dependencies(update_op):
    #    optim = tf.train.AdamOptimizer(learning_rate=learning_rates, beta1=0.9, beta2=0.999, name='AdamD').minimize(loss)

    init_op = tf.global_variables_initializer()

    var_lists=[var for var in tf.global_variables() if 'moving' in var.name]
    var_lists+=tf.trainable_variables()
    
    #----------------------------------------------------------------
    #show all trainable parameters

    #vars_list=[]
    #for va in var_lists:
    #    vars_list.append(va)
    
    #for va in var_lists:
    #    print(va)
    #----------------------------------------------------------------

    saver = tf.train.Saver(var_list=var_lists,max_to_keep=60)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth=True
    def count():
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters/1e6
    print("trainable params:", count())
    with tf.Session(config=run_config) as sess:
    
        sess.run(init_op)
        #load the ckpt
        if start_model_idx > 0:
            print('loading ckpt epoch:',start_model_idx)
            saver.restore(sess, './training_saver/model_ckpt_%d.ckpt' % (start_model_idx))

        #check whether exist necessary dirs    
        if not os.path.exists('./training_saver'):
            os.makedirs('./training_saver')
        if not os.path.exists('./training_saver/training'):
            os.makedirs('./training_saver/training')
        if not os.path.exists('./training_saver/testing'):
            os.makedirs('./training_saver/testing')  
        
        results_image_all=[]
        results_flow_all=[]
        results_fusion_all=[]
        for j in range(testing_raw.max_index):
            tt_raw=testing_raw.next_batch()
            test_iamges,test_flows=preprocess(tt_raw,dataset=dataset,h=h,w=w,c=c,gray=gray,aug=False)
               
            print(j,'------ of ------',testing_raw.max_index)
            stt_s=time.clock()
            ##,recons_images_wo_mem_details,loss_recons_flow_wo_mem_details\
            recons_images_details,s ,recons_flows_details \
                ,result_image_only_details,result_flow_only_details,\
                    result_fusion_details,loss_recons_flows_4d_details= \
                            sess.run([recons_images,sims,recons_flows\
                                ##,recons_images_wo_mem,loss_recons_flow_wo_mem\                             
                                    ,result_image_only,result_flow_only,result_fusion,\
                                        loss_recons_flows_4d],
                                        
                                        feed_dict={inputs_images: test_iamges,
                                                    inputs_flows: test_flows                                                       
                                                    })
            
            stt_e=time.clock()
            stt+=(stt_e-stt_s)

            if j % 30 == 0:   
                visulize_flow_file(loss_recons_flows_4d_details[0],'./training_saver/testing/recons_flow_error_%4d.png' %(j))                                     
                visulize_flow_file(test_flows[0],'./training_saver/testing/origin_flow_%4d.png' %(j))
                visulize_flow_file(recons_flows_details[0],'./training_saver/testing/recons_flow_%4d.png' %(j))
                cv2.imwrite('./training_saver/testing/recons_image_%4d.png' %(j),(recons_images_details[0].reshape(h,w,c)+1)*127.5)
                cv2.imwrite('./training_saver/testing/origin_image_%4d.png' %(j),(test_iamges[0].reshape(h,w,c)+1)*127.5)

            #items selected from memory per batch
            print(np.sum(s>0)) 
            
            results_image_all.append(result_image_only_details)       
            results_flow_all.append(result_flow_only_details)
            results_fusion_all.append(result_fusion_details)

        results_image_all=np.float32(results_image_all).reshape(test_samples,-1)
        results_flow_all=np.float32(results_flow_all).reshape(test_samples,-1)
        results_fusion_all=np.float32(results_fusion_all).reshape(test_samples,-1) 
        
        #----------------------------------------------------------------
        #inference procedure

        results_image_all=results_image_all-min(results_image_all)
        results_flow_all=results_flow_all-min(results_flow_all)
        results_fusion_all=results_fusion_all-min(results_fusion_all)

        image_score=results_image_all/max(results_image_all)
        flow_score=results_flow_all/max(results_flow_all)
        fusion_score=results_fusion_all/max(results_fusion_all)

        gt=np.load('gt_'+dataset+'.npy')
        print('evalutaion fps:', test_samples/stt)  # 15282 for avenue  40684 for shanghaitech

        print(' testing AUC of image only : AUC {:.2f}, EER {:.2f}',inference(gt,image_score,1))
        print(' testing AUC of flow  only : AUC {:.2f}, EER {:.2f}',inference(gt,flow_score,1))
        print(' testing AUC of   fusion   : AUC {:.2f}, EER {:.2f}',inference(gt,fusion_score,1))
        #----------------------------------------------------------------


flags=tf.app.flags
flags.DEFINE_string('dt',"ped2",'which dataset to train')
flags.DEFINE_string('g',"5",'which gpu to train')
flags.DEFINE_integer('t',0,'testing which epoch')
flags.DEFINE_integer('bs',15,'batch size')
flags.DEFINE_integer('mc',2000,'memory capacity')
flags.DEFINE_boolean('gy',False,'if gray scale')

FLAGS=flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.g

if FLAGS.dt=='ped2':
	dt=DataLoader(root_ori='/home/junwen/zfs/dataset/UCSDped2/',root_flo='/home/junwen/zfs/dataset/UCSDped2_Flow/',train=False,batch_size=FLAGS.bs,if_sub=True)
elif FLAGS.dt=='avenue':
	dt=DataLoader(root_ori='/home/junwen/zfs/dataset/Avenue/',root_flo='/home/junwen/zfs/dataset/Avenue_Flow',train=False,batch_size=FLAGS.bs,if_sub=True)
elif FLAGS.dt=='shanghaiTech':
    dt=DataLoader(root_ori='/home/junwen/zfs/dataset/shanghaiTech/',root_flo='/home/junwen/zfs/dataset/shanghaiTech_Flow',train=False,batch_size=FLAGS.bs,if_sub=True)


test(dt,start_model_idx=FLAGS.t,dataset=FLAGS.dt,mem_capacity=FLAGS.mc, batch_size=FLAGS.bs,gray=FLAGS.gy,train=False)



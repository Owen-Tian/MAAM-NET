# coding: utf-8
import os
import numpy as np
import tensorflow as tf
import cv2

from memory_module import memory
from Dataset import DataLoader,preprocess
from model import encoder,decoder,decoder_wo_mem
from flow_utils import visulize_flow_file

def cos_sim(input1,input2):

    numerator=tf.matmul(input1,input2,transpose_b=True)
    
    denominator=tf.matmul(input1**2,input2**2,transpose_b=True)

    w=(numerator+1e-12)/(denominator+1e-12)

    return w
		
def train(training_raw,max_epoch, start_model_idx=0, batch_size=18,mem_capacity=2000, lr=0.00005,dataset='ped',gray=False):
    #image size
    if gray:
        c=1
    else:
        c=3
    if dataset=='ped2':
        h,w,c=240,360,c
    elif dataset=='avenue' or dataset=='shanghaiTech':        
        h,w,c=180,320,c   
    
    #the one is for images,the other for flows  
    inputs_images=tf.placeholder(tf.float32,[batch_size,h,w,c])
    inputs_flows=tf.placeholder(tf.float32,[batch_size,h,w,2])
    
    #z is latent vector as a query to obtain latent vector for recons from memory
    z=encoder(inputs_images)

    #initialize memory
    shrink_thres=1./mem_capacity
    memo=memory(mem_capacity,z.shape[-1],shrink_thres)

    #latent is for recons for images and flows,sims is to count items numbers,sis is raw sims
    latent,sims,sis=memo.query(z)

    #obtain recons images and flows
    recons_images,recons_flows=decoder(latent,h=h,w=w,c=c)
    #recons_images_wo_mem,recons_flows_wo_mem=decoder_wo_mem(z,h=h,w=w,c=c)
    
    #if need discriminator
    #dis_true,dis_true_logits=Discriminator(inputs_images,inputs_flows)
    #dis_false,dis_false_logits=Discriminator(inputs_images,recons_flows,reuse=True)

    #l2 loss of RGBs and l1 loss of Flows 
    loss_recons_image=tf.reduce_mean(tf.square(recons_images-inputs_images))
    loss_recons_flow=tf.reduce_mean(tf.abs(recons_flows-inputs_flows))


    #loss_recons_image_wo_mem=tf.reduce_mean(tf.square(recons_images_wo_mem-inputs_images))
    #loss_recons_flow_wo_mem=tf.reduce_mean(tf.abs(recons_flows_wo_mem-inputs_flows))
    
    #find the RGB loss in the batch
    loss_recons_image_batch=tf.reduce_mean(tf.square(recons_images-inputs_images),axis=(1,2,3))
    #loss_recons_image_batch_wo_mem=tf.reduce_mean(tf.square(recons_images_wo_mem-inputs_images),axis=(1,2,3))

    #loss_grad
    #dy1, dx1 = tf.image.image_gradients(recons_images)
    #dy0, dx0 = tf.image.image_gradients(inputs_images)
    #loss_grad = tf.reduce_mean(tf.abs(tf.abs(dy1)-tf.abs(dy0)) + tf.abs(tf.abs(dx1)-tf.abs(dx0)))
    
    #Latent Loss
    #loss_latent=tf.maximum(tf.abs(tf.reduce_mean(latent-z))-0.1,0.0)
    loss_latent=tf.maximum(tf.abs(tf.reduce_mean(cos_sim(latent,z)))-0.1,0.0)
		
	#Sparsity Loss
    loss_sparsity=tf.reduce_mean((0-sims)*tf.log(sims+1e-12))
    
    #total loss
    #loss= loss_recons_image  + 0.0003*loss_sparsity + 0.001*loss_latent
    #loss= 2*loss_recons_flow + 0.0003*loss_sparsity + 0.001*loss_latent
    loss= loss_recons_image + 2*loss_recons_flow +  0.001*loss_latent+0.0003*loss_sparsity
    #loss_wo_mem=loss_recons_image_wo_mem+2*loss_recons_flow_wo_mem
    #learning_rates
    #learning_rates=tf.train.exponential_decay(learning_rate=0.0001,global_step=1000,decay_steps=100,decay_rate=1./10,staircase=True,name='ls_dc')
    learning_rates=lr

    #saves the batch moving mean and var
    update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_op):
        optim = tf.train.AdamOptimizer(learning_rate=learning_rates, beta1=0.9, beta2=0.999, name='AdamOp').minimize(loss)
        #optim_wo_mem=tf.train.AdamOptimizer(learning_rate=learning_rates, beta1=0.9, beta2=0.999, name='AdamOp2').minimize(loss_wo_mem)
    init_op = tf.global_variables_initializer()

    var_lists=[var for var in tf.global_variables() if 'moving' in var.name]
    var_lists+=tf.trainable_variables()
    
    vars_list=[]
    for va in var_lists:
        vars_list.append(va)
		
    for va in var_lists:
        print(va)
    
    saver = tf.train.Saver(var_list=var_lists,max_to_keep=100)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth=True
	
    with tf.Session(config=run_config) as sess:
        
        sess.run(init_op)
		
        #load the ckpt if needed
        if start_model_idx > 0:
            print('loading ckpt epoch:',start_model_idx)
            saver.restore(sess, './training_saver/model_ckpt_%d.ckpt' % (start_model_idx))
            print('loading ckpt successful!')	

        #check whether exist necessary dirs    
        if not os.path.exists('./training_saver'):
            os.makedirs('./training_saver')
        if not os.path.exists('./training_saver/training'):
            os.makedirs('./training_saver/training')
        if not os.path.exists('./training_saver/testing'):
            os.makedirs('./training_saver/testing')  
        
        print('learning rate is:',learning_rates)

        #loss keeper
        if os.path.exists('total_loss.npy'):
            total_loss=np.load('total_loss.npy')
        else:
            total_loss=np.float32([])
        
        #training
        for i in range(start_model_idx, max_epoch):

            training_raw.shuffle()
            tf.set_random_seed(i)
            np.random.seed(i)
            
            for j in range(training_raw.max_index):

                te_raw=training_raw.next_batch()
                for_feed_images,for_feed_flows=preprocess(te_raw,dataset=dataset,h=h,w=w,c=c,gray=gray,aug=False)

                
                _,_1,loss_detail,recons_images_details,loss_recons_image_details,s ,\
                        loss_recons_image_batch_details,recons_flows_details,\
                            loss_recons_flow_details,loss_latent_details,\
                                loss_sparsity_details= \
                            sess.run([optim, loss, recons_images,loss_recons_image,sims,\
                                        loss_recons_image_batch,recons_flows,loss_recons_flow,\
                                            loss_latent,loss_sparsity],
                         
                                             feed_dict={inputs_images : for_feed_images,
                                                        inputs_flows  : for_feed_flows                                                       
                                                        })
                print('epoch %d/%d, iter %3d/%d: loss = %.8f,loss_flow=%.8f, loss_image = %.8f,loss_sparsity=%.8f,loss_latent=%.8f'
                      % (i+1, max_epoch, j+1, training_raw.max_index, loss_detail, loss_recons_flow_details,loss_recons_image_details,loss_sparsity_details,loss_latent_details))
                #print('\t loss_wo_mem=%.8f,loss_flow_wo_mem=%.8f,loss_image_wo_mem=%.8f' %(loss_wo_mem_details,loss_recons_flow_details,loss_recons_image_details))
                total_loss=np.append(total_loss,loss_detail)
                #this is for see how much item are used from memory,when divided by z.shape[0:3]
                print('>>>>>',int(np.sum(s>0)/(batch_size*int(z.shape[1])*int(z.shape[2]))),'<<<<<')
                
                #visualize training procedure
                if j % 500 == 0:
                    visulize_flow_file(for_feed_flows[0],'./training_saver/training/origin_flow_%d_%d.png' %(i,j))
                    visulize_flow_file(recons_flows_details[0],'./training_saver/training/recons_flow_%d_%d.png' %(i,j))
                    cv2.imwrite('./training_saver/training/recons_image_%d_%d.png' %(i,j),(recons_images_details[0].reshape(h,w,c)+1)*127.5)
                    cv2.imwrite('./training_saver/training/origin_image_%d_%d.png' %(i,j),(for_feed_images[0].reshape(h,w,c)+1)*127.5)
            
            #saving ckpt
            saver.save(sess, './training_saver/model_ckpt_%d.ckpt' % (i+1)) 

            #saving training loss			
            if i%5==0 or i==max_epoch-1:          
                np.save('training_loss',np.float32(total_loss))

flags=tf.app.flags
flags.DEFINE_string('dt',"ped2",'which dataset to train')
flags.DEFINE_string('g',"5",'which gpu to train')
flags.DEFINE_integer('st',0,'start from which epoch')
flags.DEFINE_integer('mc',2000,'memory capacity')
flags.DEFINE_integer('maxes',100,'max epochs')
flags.DEFINE_float('lr',0.00005,'learning rate')
flags.DEFINE_integer('bs',15,'batch size')
flags.DEFINE_boolean('gy',False,'if gray scale')
FLAGS=flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.g

if FLAGS.dt=='ped2':
	dt=DataLoader(root_ori='/home/junwen/zfs/dataset/UCSDped2/',root_flo='/home/junwen/zfs/dataset/UCSDped2_Flow/',train=True,batch_size=FLAGS.bs,if_sub=True)
elif FLAGS.dt=='avenue':
	dt=DataLoader(root_ori='/home/junwen/zfs/dataset/Avenue/',root_flo='/home/junwen/zfs/dataset/Avenue_Flow',train=True,batch_size=FLAGS.bs,if_sub=True)
elif FLAGS.dt=='shanghaiTech':
    dt=DataLoader(root_ori='/home/junwen/zfs/dataset/shanghaiTech/',root_flo='/home/junwen/zfs/dataset/shanghaiTech_Flow',train=True,batch_size=FLAGS.bs,if_sub=True)

train(dt,max_epoch=FLAGS.maxes,start_model_idx=FLAGS.st,batch_size=FLAGS.bs,lr=FLAGS.lr,dataset=FLAGS.dt,mem_capacity=FLAGS.mc, gray=FLAGS.gy)



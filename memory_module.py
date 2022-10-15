# coding: utf-8
import os

import numpy as np
import tensorflow as tf

class memory(object):

    def __init__(self,mem_dim,fea_dim,shrink_thres=0.01):

        self.mem_dim=mem_dim
        self.fea_dim=fea_dim
        self.memory=tf.get_variable('mem',[mem_dim,fea_dim],initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.shrink_thres=shrink_thres


    def query(self,inputs):
        bts,h,w,c=int(inputs.shape[0]),int(inputs.shape[1]),int(inputs.shape[2]),int(inputs.shape[3])
        
        q=tf.reshape(inputs,(-1,c))
        ori_sims=self.cos_sim(q,self.memory)
        raw_sims=tf.nn.softmax(ori_sims)
        
        if self.shrink_thres>0:
            #add some sparsity
            te=tf.nn.relu(raw_sims-self.shrink_thres)*raw_sims
            te2=tf.abs(raw_sims-self.shrink_thres)+1e-12
            
            relu_sims=te/te2
            
            #divide by sum along axis
            relu_sims_sum=tf.reduce_sum(relu_sims,axis=-1)
            
            sims=relu_sims/(tf.expand_dims(relu_sims_sum,-1))

        latent=tf.matmul(sims,self.memory)
        
        sims=tf.reshape(sims,(-1,h*w*int(self.memory.shape[0])))

                
        latent=tf.reshape(latent,(-1,h,w,c))

        print('input shape:',inputs.shape,' mem shape:',self.memory.shape,'sims shape :',raw_sims.shape,'latent shape:',latent.shape)

        return latent,sims,raw_sims

    def cos_sim(self,input1,input2):

        numerator=tf.matmul(input1,input2,transpose_b=True)
        
        denominator=tf.matmul(input1**2,input2**2,transpose_b=True)
        
        w=(numerator+1e-12)/(denominator+1e-12)
        
        return w
        



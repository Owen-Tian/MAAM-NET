# coding: utf-8

# Code is partly from "T.-N. Nguyen and J. Meunier, “Anomaly detection in video sequence with appearance-motion correspondence,” in Proc. IEEE Int. Conf. Comput. Vis., 2019."

import tensorflow as tf
import math

def conv2d(x, out_channel, filter_size=3, stride=1, scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        in_channel = x.get_shape()[-1]
        w = tf.get_variable('w', [filter_size[0], filter_size[1], in_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        result = tf.nn.conv2d(x, w, [1, stride, stride, 1], 'SAME') + b
        if return_filters:
            return result, w, b
        return result

def conv_transpose(x, output_shape, filter_size=3, stride=2,scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [filter_size[0], filter_size[1], output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride, stride, 1])+b
        if return_filters:
            return convt, w, b
        return convt

def Discriminator(frame_true, flow_hat, is_training=True, reuse=False, return_middle_layers=False):

    def D_conv_bn_active(x, out_channel, filter_size, stride=2, training=False, bn=True, active=leaky_relu, scope=None):
        with tf.variable_scope(scope):
            d = conv2d(x, out_channel, filter_size=filter_size, stride=stride, scope='conv')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            if active is not None:
                d = active(d)
            return d

    with tf.variable_scope('discriminator') as var_scope:
        if reuse:
            var_scope.reuse_variables()

        filters = 64
        filter_size = (4, 4)

        h0 = tf.concat([frame_true, flow_hat], -1)
        h1 = D_conv_bn_active(h0, filters, filter_size, stride=2, training=is_training, bn=False, scope='dis_h1')
        h2 = D_conv_bn_active(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='dis_h2')
        h3 = D_conv_bn_active(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='dis_h3')
        h4 = D_conv_bn_active(h3, filters*8, filter_size, stride=2, training=is_training, bn=True, active=None, scope='dis_h4')

        if return_middle_layers:
            return tf.nn.sigmoid(h4), h4, [h1, h2, h3]
        return tf.nn.sigmoid(h4), h4


def encoder(input_data, is_training=True, keep_prob=0.7, return_layers=False):
    
    def G_conv_bn_relu(x, out_channel, filter_size, tag=2,stride=2, training=False, bn=True, scope=None):
        with tf.variable_scope(scope):
            d = conv2d(x, out_channel, filter_size=filter_size,stride=stride, scope='conv')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            d = tf.nn.relu(d)
            return d

    with tf.variable_scope('generator-encoder'):
        
        h0 = input_data
        filters = 64
        filter_size = (3, 3)
        
        h1 = G_conv_bn_relu(h0, filters, filter_size, stride=1, training=is_training, bn=True, scope='gen_h1')
        h2 = G_conv_bn_relu(h1, 128, filter_size, stride=2, training=is_training, bn=True, scope='gen_h2')
        h3 = G_conv_bn_relu(h2, 256, filter_size, stride=2, training=is_training, bn=True, scope='gen_h3')
        h4 = G_conv_bn_relu(h3, 512, filter_size, stride=2, training=is_training, bn=True, scope='gen_h4')
        h5 = G_conv_bn_relu(h4, 512, filter_size, stride=2, training=is_training, bn=True, scope='gen_h5')
               
        if return_layers:
            return [h5,h4, h3, h2, h1]
        return h5

def decoder(latent, is_training=True, keep_prob=0.7, return_layers=False,reuse=False,h=180,w=320,c=1):
    
    def G_deconv_bn_dr_relu_concat(layer_input, out_shape, tag=2,filter_size=(3,3), p_keep_drop=0.7, training=False, scope=None):
        with tf.variable_scope(scope):
            
            u = conv_transpose(layer_input, out_shape,filter_size=filter_size, scope='deconv')
            u = tf.layers.batch_normalization(u, training=training)
            if training:
                u = tf.nn.dropout(u, p_keep_drop)
            u = tf.nn.relu(u)

            return u

    with tf.variable_scope('generator-decoder') as v:
        b_size = tf.shape(latent)[0]

        filter_size = (3, 3)
        if reuse:
            v.reuse_variables()
        print('dropout rate:',1-keep_prob)
        latent_h1,latent_w1=math.ceil(h/2),math.ceil(w/2) #120,180   
        latent_h2,latent_w2=math.ceil(latent_h1/2),math.ceil(latent_w1/2) #60,90  
        latent_h3,latent_w3=math.ceil(latent_h2/2),math.ceil(latent_w2/2) #30,45  

        h4fl = G_deconv_bn_dr_relu_concat(latent,[b_size, latent_h3, latent_w3, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h4fl')
        h3fl = G_deconv_bn_dr_relu_concat(h4fl, [b_size, latent_h2, latent_w2, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h3fl')
        h2fl = G_deconv_bn_dr_relu_concat(h3fl, [b_size, latent_h1, latent_w1, 128],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h2fl')
        h1fl = G_deconv_bn_dr_relu_concat(h2fl, [b_size, h, w, 64],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h1fl')
        
        h0fl = conv2d(h1fl, c,filter_size=3,stride=1, scope='gen_h0fl')


        h4fr = G_deconv_bn_dr_relu_concat(latent,[b_size, latent_h3, latent_w3, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h4fr')
        h3fr = G_deconv_bn_dr_relu_concat(h4fr, [b_size, latent_h2, latent_w2, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h3fr')
        h2fr = G_deconv_bn_dr_relu_concat(h3fr, [b_size, latent_h1, latent_w1, 128],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h2fr')
        h1fr = G_deconv_bn_dr_relu_concat(h2fr, [b_size, h, w, 64],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h1fr')
        
        h0fr = conv2d(h1fr,2,filter_size=3,stride=1, scope='gen_h0fr')

        if return_layers:
            return h0fl,h2fl,h3fl,h4fl
        return h0fl,h0fr

def decoder_wo_mem(latent, is_training=True, keep_prob=0.7, return_layers=False,reuse=False,h=180,w=320,c=1):
    
    def G_deconv_bn_dr_relu_concat(layer_input, out_shape, tag=2,filter_size=(3,3), p_keep_drop=0.7, training=False, scope=None):
        with tf.variable_scope(scope):
            
            u = conv_transpose(layer_input, out_shape,filter_size=filter_size, scope='deconv')
            u = tf.layers.batch_normalization(u, training=training)
            if training:
                u = tf.nn.dropout(u, p_keep_drop)
            u = tf.nn.relu(u)

            return u

    with tf.variable_scope('generator-decoder') as v:
        b_size = tf.shape(latent)[0]

        filter_size = (3, 3)
        if reuse:
            v.reuse_variables()
        print('dropout rate:',1-keep_prob)
        latent_h1,latent_w1=h//2,w//2 #120,180
        latent_h2,latent_w2=latent_h1//2,latent_w1//2 #60,90
        latent_h3,latent_w3=latent_h2//2,latent_w2//2 #30,45

        h4fl = G_deconv_bn_dr_relu_concat(latent,[b_size, latent_h3, latent_w3, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h4flm')
        h3fl = G_deconv_bn_dr_relu_concat(h4fl, [b_size, latent_h2, latent_w2, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h3flm')
        h2fl = G_deconv_bn_dr_relu_concat(h3fl, [b_size, latent_h1, latent_w1, 128],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h2flm')
        h1fl = G_deconv_bn_dr_relu_concat(h2fl, [b_size, h, w, 64],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h1flm')
        
        h0fl = conv2d(h1fl, c,filter_size=3,stride=1, scope='gen_h0flm')


        h4fr = G_deconv_bn_dr_relu_concat(latent,[b_size, latent_h3, latent_w3, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h4frm')
        h3fr = G_deconv_bn_dr_relu_concat(h4fr, [b_size, latent_h2, latent_w2, 256],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h3frm')
        h2fr = G_deconv_bn_dr_relu_concat(h3fr, [b_size, latent_h1, latent_w1, 128],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h2frm')
        h1fr = G_deconv_bn_dr_relu_concat(h2fr, [b_size, h, w, 64],filter_size=filter_size, p_keep_drop= keep_prob, training=is_training, scope='gen_h1frm')
        
        h0fr = conv2d(h1fr,2,filter_size=3,stride=1, scope='gen_h0frm')

        if return_layers:
            return h0fl,h2fl,h3fl,h4fl
        return h0fl,h0fr

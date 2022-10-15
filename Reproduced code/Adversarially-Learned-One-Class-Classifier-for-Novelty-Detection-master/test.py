import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import pp, visualize, to_json, show_all_variables
from models import ALOCC_Model
import matplotlib.pyplot as plt
from kh_tools import *
import numpy as np
import scipy.misc
from utils import *
import time
import os
import sys

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("attention_label", 1,
                     "Conditioned label that growth attention of training label [1]")
flags.DEFINE_float("r_alpha", 0.2, "Refinement parameter [0.2]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 30, "The size of image to use. [45]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 30,
                     "The size of the output images to produce [45]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "UCSD", "The name of dataset [UCSD, mnist]")
flags.DEFINE_string(
    "dataset_address", "/home/junwen/Desktop/ALLOC/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/", "The path of dataset")
flags.DEFINE_string("input_fname_pattern", "*",
                    "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "optim_ckpt",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("sample_dir", "samples",
                    "Directory name to save the image samples [samples]")
flags.DEFINE_boolean(
    "train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS


def check_some_assertions():
    """
    to check some assertions in inputs and also check sth else.
    """
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)


def main(_):
    print('Program is started at', time.clock())
    pp.pprint(flags.FLAGS.__flags)

    n_per_itr_print_results = 100
    n_fetch_data = 180
    kb_work_on_patch = False
    nd_input_frame_size = (240, 360)
    #nd_patch_size = (45, 45)
    n_stride = 10
    nd_patch_step = (n_stride, n_stride)
    #FLAGS.checkpoint_dir = "./checkpoint/UCSD_128_45_45/"
    #FLAGS.dataset = 'UCSD'
    #FLAGS.dataset_address = './dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

    check_some_assertions()

    nd_patch_size = (FLAGS.input_width, FLAGS.input_height)
    # FLAGS.nStride = n_stride

    #FLAGS.input_fname_pattern = '*'
    FLAGS.train = False
    FLAGS.epoch = 1
    FLAGS.batch_size = 56

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        tmp_ALOCC_model = ALOCC_Model(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            attention_label=FLAGS.attention_label,
            r_alpha=FLAGS.r_alpha,
            is_training=FLAGS.train,
            dataset_name=FLAGS.dataset,
            dataset_address=FLAGS.dataset_address,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            nd_patch_size=nd_patch_size,
            n_stride=n_stride,
            n_per_itr_print_results=n_per_itr_print_results,
            kb_work_on_patch=kb_work_on_patch,
            nd_input_frame_size=nd_input_frame_size,
            n_fetch_data=n_fetch_data)

        show_all_variables()

        print('--------------------------------------------------')
        print('Load Pretrained Model...')
        tmp_ALOCC_model.f_check_checkpoint()

            #generated_data = tmp_ALOCC_model.feed2generator(data[0:FLAGS.batch_size])

        # else in UCDS (depends on infrustructure)
        tmp_lst_image_paths = []
        tmp_gt = []

        # append the directories in the list you want to test
        for s_image_dirs in sorted(glob(os.path.join(FLAGS.dataset_address, 'Test[0-9][0-9][0-9]'))):

            #if os.path.basename(s_image_dirs) not in ['Test004']:
            #    print('Skip ', os.path.basename(s_image_dirs))
            #    continue
            for s_image_dir_files in sorted(glob(os.path.join(s_image_dirs + '/*'))):
                tmp_lst_image_paths.append(s_image_dir_files)

        # append the ground truth directories
        for s_image_dirs in sorted(glob(os.path.join(FLAGS.dataset_address, 'Test[0-9][0-9][0-9]_gt'))):

            #if os.path.basename(s_image_dirs) not in ['Test004_gt']:
            #    print('Skip ', os.path.basename(s_image_dirs))
            #    continue
            for s_image_dir_files in sorted(glob(os.path.join(s_image_dirs + '/*'))):
                tmp_gt.append(s_image_dir_files)


        lst_image_paths = tmp_lst_image_paths

        images = read_lst_images_without_noise2(
            lst_image_paths, nd_patch_size, nd_patch_step)

        lst_prob = process_frame(images, tmp_gt, tmp_ALOCC_model)

        print('Test is finished')



def process_frame(frames_src, tmp_gt, sess):


    errors=0
    anom_count=0
    nd_patch, nd_location = get_image_patches(
        frames_src, sess.patch_size, sess.patch_step)

    print(np.array(nd_patch).shape)
    frame_patches = nd_patch.transpose([1, 0, 2, 3])
    
    print('frame patches :{}\npatches size:{}'.format(
        len(frame_patches[0]), (frame_patches.shape[2], frame_patches.shape[3])))
    lst_prob = sess.f_test_frozen_model(frame_patches)

    #iterate over every frame
    for index in range(len(frames_src)):

        count = 0
        anomaly = np.zeros((240, 360, 3))
        lst_anomaly = []
        for i in nd_location:
            # for every patch check if probability < 0.3 and skip the bottom right patches (always anomaly due to unusual grass patch)
            if lst_prob[index][count] < 0.3 and i[0]!=110 and i!=[30,300] and i!=[80,320] and i!=[80,330]:
                lst_anomaly.append(i)
                for j in range(30):
                    for k in range(30):
                        for l in range(3):
                            anomaly[100 + i[0] + j][i[1] + k][l] += 1
            count += 1
        print(lst_anomaly)

        # make the anomaly matrix binary 0->normal 1->anomaly
        for i in range(240):
            for j in range(360):
                for k in range(3):
                    if(anomaly[i][j][k] < 1):
                        anomaly[i][j][k] = 0
                    else:
                        anomaly[i][j][k] = 1

        plt.imsave(arr=anomaly, vmin=0, vmax=1, fname="anomalies/anomaly"+str(index)+".jpg")
        temp = scipy.misc.imread(tmp_gt[index])
        # check if frame anomaly and ground_truth anomaly and if mismatch add to errors
        if np.sum(anomaly)>0:
            anom_count+=1
        if np.sum(temp)==0 and np.sum(anomaly)>0:
            errors+=1
        if np.sum(anomaly)==0 and np.sum(temp)>0:
            errors+=1
    
    
    print("No. of anomaly frames",)
    print(anom_count)
    print("Equal Error Rate: ",)
    print(100*errors/len(frames_src))
        # exit()



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    tf.app.run()

from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
import re
from ops import *
from utils import *
from kh_tools import *
import logging
import matplotlib.pyplot as plt
import sys
from random import shuffle


class ALOCC_Model(object):
    def __init__(self, sess,
                 input_height=45, input_width=45, output_height=64, output_width=64,
                 batch_size=64, sample_num=64, attention_label=1, is_training=True,
                 z_dim=100, gf_dim=64, df_dim=64, gfc_dim=512, dfc_dim=512, c_dim=1,
                 dataset_name=None, dataset_address=None, input_fname_pattern=None,
                 checkpoint_dir=None, log_dir=None, sample_dir=None, r_alpha=0.4,
                 kb_work_on_patch=True, nd_input_frame_size=(240, 360), nd_patch_size=(10, 10), n_stride=1,
                 n_fetch_data=10, n_per_itr_print_results=500):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection
        :param sess: TensorFlow session
        :param batch_size: The size of batch. Should be specified before training. [128]
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param r_alpha: Refinement parameter [0.2]
        :param z_dim:  (optional) Dimension of dim for Z. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        :param df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        :param gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        :param dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param sample_dir: Directory address which save some samples [.]
        :param kb_work_on_patch: Boolean value for working on PatchBased System or not [True]
        :param nd_input_frame_size:  Input frame size
        :param nd_patch_size:  Input patch size
        :param n_stride: PatchBased data preprocessing stride
        :param n_fetch_data: Fetch size of Data
        :param n_per_itr_print_results: # of printed iteration
        """

        self.n_per_itr_print_results = n_per_itr_print_results
        self.nd_input_frame_size = nd_input_frame_size
        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir

        self.sess = sess
        self.is_training = is_training

        self.r_alpha = r_alpha

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn0 = batch_norm(name='d_bn0', train=self.is_training)
        self.d_bn1 = batch_norm(name='d_bn1', train=self.is_training)
        self.d_bn2 = batch_norm(name='d_bn2', train=self.is_training)
        self.d_bn3 = batch_norm(name='d_bn3', train=self.is_training)

        self.g_bn0 = batch_norm(name='g_bn0', train=self.is_training)
        self.g_bn1 = batch_norm(name='g_bn1', train=self.is_training)
        self.g_bn2 = batch_norm(name='g_bn2', train=self.is_training)
        self.g_bn3 = batch_norm(name='g_bn3', train=self.is_training)
        self.g_bn4 = batch_norm(name='g_bn4', train=self.is_training)
        self.g_bn5 = batch_norm(name='g_bn5', train=self.is_training)
        self.g_bn6 = batch_norm(name='g_bn6', train=self.is_training)
        self.g_bn7 = batch_norm(name='g_bn7', train=self.is_training)

        self.dataset_name = dataset_name
        self.dataset_address = dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.attention_label = attention_label

        if self.is_training:
            logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)


        if self.dataset_name == 'UCSD':
            self.nStride = n_stride
            self.patch_size = nd_patch_size
            self.patch_step = (n_stride, n_stride)
            # print(self.__dict__)
            lst_image_paths = []
            for s_image_dir_path in glob(os.path.join(self.dataset_address, self.input_fname_pattern)):
                for sImageDirFiles in glob(os.path.join(s_image_dir_path + '/*')):
                    lst_image_paths.append(sImageDirFiles)
            self.dataAddress = lst_image_paths
            lst_forced_fetch_data = [self.dataAddress[x] for x in random.sample(
                range(0, len(lst_image_paths)), n_fetch_data)]
            self.data = lst_forced_fetch_data
            self.c_dim = 1
            self.grayscale = 1
        else:
            assert('Error in loading dataset')

        self.grayscale = (self.c_dim == 1)
        self.build_model()

    # =========================================================================================================
    def build_model(self):

        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs


        # The error function added to the image
        self.z = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='z')

        # Generate the Images
        self.G, self.G_ = self.generator(self.z)

        # Generator apply sigmoid
        # Use the Discriminator to discriminate for the generator's image and the input image

        self.D, self.D_logits = self.discriminator(inputs)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)


        # Simple GAN's losses
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits, labels=tf.zeros_like(self.D)))


        # Non Saturating Heuristic
        self.g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        self.g_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits, labels=tf.zeros_like(self.D)))


        # Refinement loss
        self.g_r_loss = tf.reduce_mean(tf.square(self.G_ - self.z))


        # Final Loss
        self.g_loss = self.g_loss_real + \
            self.g_loss_fake + (self.g_r_loss * self.r_alpha)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_loss_real_sum = tf.summary.scalar(
            "d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar(
            "d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]


# =========================================================================================================

    def train(self, config):

        # Discriminator and generator optimizers
        d_grads = tf.train.AdamOptimizer(config.learning_rate).compute_gradients(
            self.d_loss, var_list=self.d_vars)
        g_grads = tf.train.AdamOptimizer(config.learning_rate).compute_gradients(
            self.g_loss, var_list=self.g_vars)

        d_optim = tf.train.AdamOptimizer(
            config.learning_rate).apply_gradients(d_grads)
        g_optim = tf.train.AdamOptimizer(
            config.learning_rate).apply_gradients(g_grads)


        d_grads_scalar = []
        g_grads_scalar = []
        for grad, var in d_grads:
            d_grads_scalar.append(tf.summary.histogram(
                var.name + '/d_gradient', grad))
        for grad, var in g_grads:
            g_grads_scalar.append(tf.summary.histogram(
                var.name + '/g_gradient', grad))
        # Merge all summaries into a single op

        print(d_grads_scalar)
        print(g_grads_scalar)
        self.d_grads = merge_summary(d_grads_scalar)
        self.g_grads = merge_summary(g_grads_scalar)
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=40)

        self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])

        log_dir = os.path.join(self.log_dir, self.model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir, self.sess.graph)

        # creating a standard sample to compare results after some fixed interval
        if config.dataset == 'UCSD':
            if self.b_work_on_patch:
                sample_files = self.data[0:10]
            else:
                sample_files = self.data[0:self.sample_num]

            sample, _ = read_lst_images_w_noise(
                sample_files, self.patch_size, self.patch_step)
            sample = np.array(sample).reshape(-1,
                                              self.patch_size[0], self.patch_size[1], 1)

            sample = sample[0:self.sample_num]

        # export images
        sample_inputs = np.array(sample).astype(np.float32)
        scipy.misc.imsave('./{}/train_input_samples.jpg'.format(
            config.sample_dir), montage(sample_inputs[:, :, :, 0]))
        # # load previous checkpoint
        counter = 1
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load traning data

        # Extract image patches from sample data
        if config.dataset == 'UCSD':
            sample_files = self.data
            sample, _ = read_lst_images(
                sample_files, self.patch_size, self.patch_step, self.b_work_on_patch)
            sample = np.array(sample).reshape(-1,
                                              self.patch_size[0], self.patch_size[1], 1)
            sample_w_noise, _ = read_lst_images_w_noise(
                sample_files, self.patch_size, self.patch_step)
            sample_w_noise = np.array(
                sample_w_noise).reshape(-1, self.patch_size[0], self.patch_size[1], 1)

        for epoch in xrange(config.epoch):
            print(
                'Epoch ({}/{})-------------------------------------------------'.format(epoch, config.epoch))
            if config.dataset == 'UCSD':
                batch_idxs = min(
                    len(sample), config.train_size) // config.batch_size

            # for detecting valuable epoch that we must stop training step
            # sample_input_for_test_each_train_step.npy
            # sample_test = np.load('SIFTETS.npy').reshape(
            #     [504, 45, 45, 1])[0:64]
            sample_test = sample_inputs

            for idx in xrange(0, batch_idxs):

                if config.dataset == 'UCSD':
                    batch = sample[idx *
                                   config.batch_size:(idx + 1) * config.batch_size]
                    batch_noise = sample_w_noise[idx *
                                                 config.batch_size:(idx + 1) * config.batch_size]

                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)

                # batch_z = np.random.uniform(
                #     0, 1, [config.batch_size, self.z_dim]).astype(np.float32)


                # Run generator twice and discriminator once for a balanced min max game
                self.sess.run(g_optim, feed_dict={
                              self.inputs: batch_images, self.z: batch_noise_images})
                self.sess.run(g_optim, feed_dict={
                              self.inputs: batch_images, self.z: batch_noise_images})

                summary_str, grads, a, b, g = self.sess.run([self.g_sum, self.g_grads, self.D, self.D_, self.G],
                                                            feed_dict={self.inputs: batch_images, self.z: batch_noise_images})
                self.writer.add_summary(summary_str, counter)
                self.writer.add_summary(grads, counter)
                self.writer.flush()

                self.sess.run(d_optim, feed_dict={
                              self.inputs: batch_images, self.z: batch_noise_images})
                summary_str, grads, c, d = self.sess.run([self.d_sum, self.d_grads, self.D, self.D_],
                                                         feed_dict={self.inputs: batch_images, self.z: batch_noise_images})
                self.writer.add_summary(summary_str, counter)
                self.writer.add_summary(grads, counter)
                self.writer.flush()

                errD_fake = self.d_loss_fake.eval(
                    {self.inputs: batch_images, self.z: batch_noise_images})
                errD_real = self.d_loss_real.eval(
                    {self.inputs: batch_images, self.z: batch_noise_images})
                errG_real = self.g_loss_real.eval(
                    {self.inputs: batch_images, self.z: batch_noise_images})
                errG_fake = self.g_loss_fake.eval(
                    {self.inputs: batch_images, self.z: batch_noise_images})
                errG_r = self.g_r_loss.eval(
                    {self.inputs: batch_images, self.z: batch_noise_images})
                counter += 1

                # Print losses and probabilities after every 10 batches
                if idx % 10 == 0:
                    msg = "Epoch:[%2d][%4d/%4d]--> Fake_d_loss: %.8f, Real_d_loss: %.8f\n Fake_g_loss: %.8f, Real_g_loss: %.8f, gr_loss: %.8f\n" % (
                        epoch, idx, batch_idxs, errD_fake, errD_real, errG_fake, errG_real, errG_r)
                    print(msg)
                    logging.info(msg)
                    msg = "After Generator: Real D Prob: %.8f, Fake D Prob: %.8f\n" % (
                        np.mean(a), np.mean(b))
                    print(msg)
                    logging.info(msg)

                    msg = "After Discriminator: Real D Prob: %.8f, Fake D Prob: %.8f\n" % (
                        np.mean(c), np.mean(d))
                    print(msg)
                    logging.info(msg)

                if np.mod(counter, self.n_per_itr_print_results) == 0:
                    # test on sample after every n_per_itr_print_results batches
                    # ====================================================================================================
                    samples, d_loss, g_loss, a = self.sess.run(
                        [self.G, self.d_loss, self.g_loss, self.D_],
                        feed_dict={
                            self.z: sample_inputs,
                            self.inputs: sample_inputs,
                        },
                    )

                    scipy.misc.imsave('./{}/z_test_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),
                                      montage(samples[:, :, :, 0]))
                    scipy.misc.imsave('./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),
                                      montage(g[:, :, :, 0]))

                    msg = "[Sample] d_loss: %.8f, g_loss: %.8f" % (
                        d_loss, g_loss)
                    print(msg)
                    logging.info(msg)
                    msg = "D(R(x)): %.8f" % (np.mean(a))
                    print(msg)
                    logging.info(msg)

            # Test on a testing data image after every epoch
            test_img = read_lst_images_w_noise2(
                ["/home/junwen/Desktop/ALLOC/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004/068.tif"], (30, 30), (10, 10))
            test_img_orig = read_lst_images_without_noise2(
                ["/home/junwen/Desktop/ALLOC/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004/068.tif"], (30, 30), (10, 10))

            test_patch, test_location = get_image_patches(
                test_img, (30, 30), (10, 10))
            test_patch_orig, test_location_orig = get_image_patches(
                test_img_orig, (30, 30), (10, 10))

            frame_test_patches = np.expand_dims(
                np.squeeze(test_patch), axis=-1)
            frame_test_patches_orig = np.expand_dims(
                np.squeeze(test_patch_orig), axis=-1)

            test_batch_idxs = len(frame_test_patches) // self.batch_size
            print(len(frame_test_patches))
            print('start new process ...')
            lst_generated_img = []
            lst_discriminator_v = []
            # append generated images and probabilities to separate lists
            for i in xrange(0, test_batch_idxs):
                batch_data = frame_test_patches[i *
                                                self.batch_size:(i + 1) * self.batch_size]

                results_g = self.sess.run(
                    self.G, feed_dict={self.z: batch_data})
                results_d = self.sess.run(
                    self.D_, feed_dict={self.z: batch_data})

                lst_discriminator_v.extend(results_d)
                lst_generated_img.extend(results_g)
                print('finish pp ... {}/{}'.format(i, test_batch_idxs))


            start = (test_batch_idxs - 1) * self.batch_size
            end = (test_batch_idxs) * self.batch_size
            errD_fake = self.d_loss_fake.eval(
                {self.inputs: frame_test_patches_orig[start:end], self.z: frame_test_patches[start:end]})
            errD_real = self.d_loss_real.eval(
                {self.inputs: frame_test_patches_orig[start:end], self.z: frame_test_patches[start:end]})
            errG = self.g_loss.eval(
                {self.inputs: frame_test_patches_orig[start:end], self.z: frame_test_patches[start:end]})
            msg = "Fake_d_loss: %.8f, Real_d_loss: %.8f, g_loss: %.8f" % (
                errD_fake, errD_real, errG)
            print(msg)

            scipy.misc.imsave('./' + self.sample_dir + '/ALOCC_generated' + str(
                epoch) + '.jpg', montage(np.array(lst_generated_img)[:, :, :, 0]))
            scipy.misc.imsave('./' + self.sample_dir + '/ALOCC_input' + str(epoch) +
                              '.jpg', montage(np.array(frame_test_patches)[:, :, :, 0]))
            self.save(config.checkpoint_dir, epoch)

    # =========================================================================================================
    def discriminator(self, image, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            # df_dim = 64 (data frame dimension)
            h0 = lrelu(self.d_bn0(
                conv2d(image, self.df_dim, name='d_h0_conv')))
            assert(h0.get_shape()[-1] == 64)

            h1 = lrelu(self.d_bn1(
                conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            assert(h1.get_shape()[-1] == 128)

            h2 = lrelu(self.d_bn2(
                conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            assert(h2.get_shape()[-1] == 256)

            h3 = lrelu(self.d_bn3(
                conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            assert(h3.get_shape()[-1] == 512)

            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            h5 = tf.nn.sigmoid(h4, name='d_output')

            return h5, h4

    # =========================================================================================================
    def generator(self, z, reuse=None):
        ''' Convolution Auto-Encoder Decoder '''

        with tf.variable_scope("generator", reuse=reuse) as scope:

            # Encoder-Architecture
            # gf_dim = 64 (data frame dimension)
            encoder_0 = lrelu(self.g_bn0(
                conv2d(z, self.gf_dim, name='g_encoder_h0_conv')), 0)
            assert(encoder_0.get_shape()[-1] == 64)

            encoder_1 = lrelu(self.g_bn1(
                conv2d(encoder_0, self.gf_dim * 2, name='g_encoder_h1_conv')), 0)
            assert(encoder_1.get_shape()[-1] == 128)

            encoder_2 = lrelu(self.g_bn2(
                conv2d(encoder_1, self.gf_dim * 4, name='g_encoder_h2_conv')), 0)
            assert(encoder_2.get_shape()[-1] == 256)

            # Middle Portion
            encoder_3 = lrelu(self.g_bn3(
                conv2d(encoder_2, self.gf_dim * 8, name='g_encoder_h3_conv', padding='SAME')), 0)
            assert(encoder_3.get_shape()[-1] == 512)

            print(encoder_3)
            # Decoder-Architecture
            decoder_3 = lrelu(self.g_bn4(deconv2d(encoder_3, output_shape=encoder_2.get_shape(
            ), name='g_decoder_h3_deconv', padding='SAME')), 0)
            assert(decoder_3.get_shape()[-1] == 256)
            print(decoder_3)

            decoder_2 = lrelu(self.g_bn5(deconv2d(
                decoder_3, output_shape=encoder_1.get_shape(), name='g_decoder_h2_deconv')), 0)
            assert(decoder_2.get_shape()[-1] == 128)
            print(decoder_2)

            decoder_1 = lrelu(self.g_bn6(deconv2d(
                decoder_2, output_shape=encoder_0.get_shape(), name='g_decoder_h1_deconv')), 0)
            assert(decoder_1.get_shape()[-1] == 64)
            print(decoder_1)

            decoder_0 = self.g_bn7(
                deconv2d(decoder_1, output_shape=z.get_shape(), name='g_decoder_h0_deconv'))
            assert(decoder_0.get_shape()[-1] == 1)
            print(decoder_0)

            return tf.nn.tanh(decoder_0, name='g_output'), decoder_0

    # =========================================================================================================
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    # =========================================================================================================
    def save(self, checkpoint_dir, step):
        model_name = "ALOCC_Model.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    # =========================================================================================================
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # =========================================================================================================

    def f_check_checkpoint(self):
        self.saver = tf.train.Saver()
        print(" [*] Success to read checkpoint")

        self.saver.restore(
            self.sess, "./optim_ckpt/ALOCC_Model.model-23")
        return 1


    # =========================================================================================================
    def f_test_frozen_model(self, lst_image_slices=[]):
        lst_generated_img = []
        lst_discriminator_v = []
        tmp_shape = lst_image_slices.shape
        tmp_lst_slices = np.reshape(
            lst_image_slices, [-1, lst_image_slices.shape[-2], lst_image_slices.shape[-1], 1])
        print(tmp_lst_slices.shape)

        batch_idxs = len(tmp_lst_slices) // self.batch_size

        print('start new process ...')
        for i in xrange(0, batch_idxs):
            batch_data = tmp_lst_slices[i *
                                        self.batch_size:(i + 1) * self.batch_size]

            results_d = self.sess.run(self.D_, feed_dict={self.z: batch_data})
            lst_discriminator_v.extend(results_d)

            print('finish pp ... {}/{}'.format(i, batch_idxs))
        lst_discriminator_v = np.reshape(np.array(lst_discriminator_v), [
                                         lst_image_slices.shape[0], -1])

        return lst_discriminator_v

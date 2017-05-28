import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os

class Inpaint():
    def __init__(self):

        self.img_size = 64
        self.num_colors = 3

        self.batch_size = 64

        self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])
        self.broken_images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])

        d_bn1 = batch_norm(name='dg_bn1')
        d_bn2 = batch_norm(name='dg_bn2')
        d_bn3 = batch_norm(name='dg_bn3')

        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')
        g_bn4 = batch_norm(name='g_bn4')

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        # breaking down the context
        h0 = lrelu(conv2d(self.broken_images, self.num_colors, 64, name='dg_h0_conv')) #32x32x64
        h1 = lrelu(d_bn1(conv2d(h0, 64, 128, name='dg_h1_conv'))) #16x16x128
        h2 = lrelu(d_bn2(conv2d(h1, 128, 256, name='dg_h2_conv'))) #8x8x256
        h2_2 = lrelu(d_bn3(conv2d(h2, 256, 256, name='dg_h3_conv'))) #4x4x256

        # generating the new replacement
        h3 = tf.nn.relu(g_bn0(tf.reshape(h2_2, [-1, 4, 4, 256]))) # 4x4x256
        h4 = tf.nn.relu(g_bn2(conv_transpose(h3, [self.batch_size, 8, 8, 128], "g_h4"))) #8x8x128
        h5 = tf.nn.relu(g_bn3(conv_transpose(h4, [self.batch_size, 16, 16, 128], "g_h5"))) #16x16x128
        self.generated_images = tf.nn.tanh(g_bn4(conv_transpose(h5, [self.batch_size, 32, 32, 3], "g_h6"))) #32x32x3

        self.genfull = self.images
        self.genfull -= tf.pad(self.images[:, 16:48, 16:48, :], [[0,0],[16,16],[16,16],[0,0]], "CONSTANT")
        self.genfull += tf.pad(self.generated_images, [[0,0],[16,16],[16,16],[0,0]], "CONSTANT")


        def discriminator(image, reuse=False):
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            self.df_dim = 64
            h0 = lrelu(conv2d(image, 3, self.df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim, self.df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*2, self.df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*4, self.df_dim*8, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
            h4 = dense(tf.reshape(h3, [self.batch_size, -1]), 8192, 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4


        self.disc_true, disc_true_logits = discriminator(self.images, reuse=False)
        self.disc_fake, disc_fake_logits = discriminator(self.genfull, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_true_logits, tf.ones_like(disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_logits, tf.zeros_like(disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_logits, tf.ones_like(disc_fake_logits))) \
                        + 100 * tf.reduce_mean(tf.abs(self.images - self.genfull))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        # self.train_op = self.cost

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def train(self):
        data = glob(os.path.join("../Datasets/celebA", "*.jpg"))
        base = np.array([get_image(sample_file, 108, is_crop=True) for sample_file in data[0:64]])
        print base.shape
        base += 1
        base /= 2
        broken_base = np.copy(base)
        broken_base[:,16:16+32,16:16+32,:] = 0
        ims("results/base.jpg",merge_color(base,[8,8]))

        #
        for e in xrange(50):
            for i in range(min(2370,  len(data) / self.batch_size)):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file, 32, is_crop=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2
                broken_images = np.copy(batch_images)
                broken_images[:,16:16+32,16:16+32,:] = 0

                dloss, _ = self.sess.run([self.d_loss, self.d_optim], feed_dict={self.images: batch_images, self.broken_images: broken_images})
                gloss, _ = self.sess.run([self.g_loss, self.g_optim], feed_dict={self.images: batch_images, self.broken_images: broken_images})
                print "%f, %f" % (dloss, gloss)
                if (i % 30 == 0) or (gloss > 10000):
                    fill = self.sess.run(self.genfull, feed_dict={self.images: batch_images, self.broken_images: broken_images})
                    ims("results/"+str(e*10000 + i)+".jpg",merge_color(fill,[8,8]))
                    ims("results/"+str(e*10000 + i)+"-base.jpg",merge_color(fill,[8,8]))



model = Inpaint()
model.train()

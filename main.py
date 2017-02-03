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

        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')
        d_bn3 = batch_norm(name='d_bn3')

        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')
        g_bn4 = batch_norm(name='g_bn4')

        # breaking down the context
        h0 = lrelu(conv2d(self.broken_images, self.num_colors, 64, name='d_h0_conv')) #32x32x64
        h1 = lrelu(d_bn1(conv2d(h0, 64, 128, name='d_h1_conv'))) #16x16x128
        h2 = lrelu(d_bn2(conv2d(h1, 128, 256, name='d_h2_conv'))) #8x8x256
        h2_2 = lrelu(d_bn3(conv2d(h2, 256, 256, name='d_h3_conv'))) #4x4x256

        bridge = dense(tf.reshape(h2_2, [self.batch_size, -1]), 4*4*256, 4*4*256, scope='d_h3_lin') #256

        # generating the new replacement
        h3 = tf.nn.relu(g_bn0(tf.reshape(bridge, [-1, 4, 4, 256]))) # 4x4x256
        h4 = tf.nn.relu(g_bn2(conv_transpose(h3, [self.batch_size, 8, 8, 128], "g_h4"))) #8x8x128
        h5 = tf.nn.relu(g_bn3(conv_transpose(h4, [self.batch_size, 16, 16, 128], "g_h5"))) #16x16x128
        self.generated_images = tf.nn.tanh(g_bn4(conv_transpose(h5, [self.batch_size, 32, 32, 3], "g_h6"))) #32x32x3
        self.target = tf.slice(self.images,[0, 16, 16, 0], [self.batch_size, 32, 32, 3])

        self.generation_loss = tf.nn.l2_loss(self.target - self.generated_images)

        self.cost = self.generation_loss
        optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g,5),v)
        self.train_op = optimizer.apply_gradients(grads)

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
        for e in xrange(20000):
            for i in range(len(data) / self.batch_size):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file, 32, is_crop=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2
                broken_images = np.copy(batch_images)
                broken_images[:,16:16+32,16:16+32,:] = 0

                fill, gen_loss, _ = self.sess.run([self.generated_images, self.generation_loss, self.train_op], feed_dict={self.images: batch_images, self.broken_images: broken_images})
                print "iter %d genloss %f" % (i, gen_loss)
                if (i % 30 == 0) or (gen_loss > 10000):
                    fill = self.sess.run(self.generated_images, feed_dict={self.images: batch_images, self.broken_images: broken_images})
                    recreation = np.copy(batch_images)
                    recreation[:,16:16+32,16:16+32,:] = fill
                    ims("results/"+str(e*10000 + i)+".jpg",merge_color(recreation,[8,8]))
                    ims("results/"+str(e*10000 + i)+"-base.jpg",merge_color(batch_images,[8,8]))



model = Inpaint()
model.train()

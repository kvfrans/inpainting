import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os

class Inpaint():
    def __init__(self):

        self.img_size = 32
        self.num_colors = 3

        self.batch_size = 1

        self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])

        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')

        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')

        # breaking down the context
        h0 = lrelu(conv2d(self.images, self.num_colors, 64, name='d_h0_conv')) #16x16x64
        h1 = lrelu(d_bn1(conv2d(h0, 64, 128, name='d_h1_conv'))) #8x8x128
        h2 = lrelu(d_bn2(conv2d(h1, 128, 256, name='d_h2_conv'))) #4x4x256

        bridge = dense(tf.reshape(h2, [self.batch_size, -1]), 4*4*256, 4*4*256, scope='d_h3_lin') #256

        # generating the new replacement
        h3 = tf.nn.relu(g_bn0(tf.reshape(bridge, [-1, 4, 4, 256]))) # 2x2x256
        h5 = tf.nn.relu(g_bn2(conv_transpose(h3, [self.batch_size, 8, 8, 128], "g_h5"))) #8x8x128
        self.generated_images = tf.nn.tanh(g_bn3(conv_transpose(h5, [self.batch_size, 16, 16, 3], "g_h6"))) #8x8x128

        self.target = tf.slice(self.images,[0, 8, 8, 0], [self.batch_size, 16, 16, 3])

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
        # data = glob(os.path.join("../Datasets/celebA", "*.jpg"))
        # base = np.array([get_image(sample_file, 108, is_crop=True) for sample_file in data[0:64]])
        # base += 1
        # base /= 2
        #
        # ims("results/base.jpg",merge_color(base,[8,8]))

        data = glob(os.path.join("data", "*.jpg"))
        base = np.array([get_image(sample_file, 32, is_crop=False) for sample_file in data[0:1]])
        base += 1
        base /= 2

        ims("results/base.jpg",base[0])

        for e in xrange(20000):
            for i in range(len(data) / self.batch_size):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file, 32, is_crop=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2

                fill, gen_loss, _ = self.sess.run([self.generated_images, self.generation_loss, self.train_op], feed_dict={self.images: batch_images})
                print "iter %d genloss %f" % (e, gen_loss)
                if e % 3 == 0:
                    recreation = batch_images
                    recreation[:,8:8+16,8:8+16,:] = fill
                    ims("results/"+str(e)+".jpg",recreation[0])



model = Inpaint()
model.train()

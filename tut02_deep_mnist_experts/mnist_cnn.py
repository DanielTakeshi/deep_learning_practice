"""
There doesn't seem to be a script for this in their GitHub so I'll just make one
here. The convolutions are not as hard as they could be because the number of
channels is one, rather than the usual three. See detailed comments below. The
outcome of my code is:

    test accuracy 0.9918

    real    1m10.899s
    user    1m27.124s
    sys     0m12.244s

So yes, I get the accuracy as suggested by the write-up. Wow, my computer is
really good ... 1 minute, 10 seconds to run this, and the write-up suggested it
could have taken a hour for all 20k iterations.
"""

from __future__ import print_function
import tensorflow as tf
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None


def weight_variable(shape):
    """ Initialize ReLUs using a 'truncated normal', not quite state of the art
    but this will be just fine for now. """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """ Simply initialize biases to 0.1 everywhere. """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """ Don't try implementing conv2d. It's done in a kernel (in C++) and is
    extremely hard to understand because it's been highly-optimized. It assumes
    the input, x, is a 4-D tensor of (batch, height, width, channels) shape. The
    padding is the same so it pads just enough to make the output layer the same
    shape. The strides should be [1, stride1, stride2, 1] where usually stride1
    and stride2 are equal. W is the filter. """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    """ ksize has size of window for each tensor dimension, which here we only
    care about index 1 and index 2 so it's a 2x2 window in one channel. """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


if __name__ == "__main__":
    """
    We have to reshape x because before it was shaped [None,784]. The first
    dimension is the batch size but we can leave the -1 as a placeholder for a
    size to be determined later.

    For the convolution variables, the last dimension is _not_ batch size
    (remember, these are weights, not data) but rather the number of _output_
    channels. The second-to-last dimension is the number of _input_ channels.

    Here's how the dimensions work (bs = batch_size):

    (1) input, (bs,28,28,1) // last argument is number of channels
    (2) after conv, (bs,28,28,32) // 32 output "channels"
    (3) after max-pool, (bs,14,14,32)
    (4) after second conv, (bs,14,14,64) // 32 input, 64 output
    (5) after second max-pool, (bs,7,7,64)
    (6) now input is (bs,7*7*64) so 2-dimensional, so FC matrix can be a 2-D
            matrix of dimension (7*7*64,1024), resulting in (bs,1024). Hence,
            each image in the batch has now been "reduced" to 1024 features.
    (7) finally, (bs,1024) turns into (bs,10) via softmax.
    """

    # Input, same as the simple example.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Now get to the CNN.
    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Now we can train and test. Looks like we use accuracy.eval now, before we
    # just ran a session with accuracy as input ...
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(
        feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

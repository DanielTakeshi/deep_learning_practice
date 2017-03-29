"""
This is from the TensorFlow tutorial, and I'm adding my comments. It assumes
version 0.12.1. I'm SO GLAD there's a `pip install tensorflow-gpu` command now.
NEVER build from source again!

The tutorial's explanation of softmax regression for MNIST is very nice and easy
to understand. Thanks for the clarity! Just do softmax(Wx + b). The TF tutorial
emphasizes that we could use numpy commands for computation, but this has the
cost of Python-to-C++ and back transfers (I'm assuming that C++ is the language
used for the "highly efficient math code.") I really should know more about
this.  Fortunately, TF says it defines a _computational_graph_ which lies
_outside_ of Python. This is the same approach used in Theano. Thus, both treat
Python as a meta-programming language since the actual computation is not done
in Python.  Also, I'm not sure if _computational_graph_ is the correct term,
perhaps a better one is a _symbolic_graph_.

The problem with this tutorial is that they provide us with an input_data
_specifically_ for MNIST. It would be better if I could just take a numpy array
and do the import right here. I hope I can figure out how to do that later. The
first time we run this, it officially loads the data from somewhere and saves it
in a /tmp/ directory so we can use it quickly later.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def main(_):
    """
    Note how we don't actually call main the usual way, but instead input it via
    tf.app.run with 'main' as an argument. The input_data.read_data_sets returns
    a 'Dataset' class object. And remember that, like in Theano, x is a
    placeholder or symbol, _not_ a normal Python variable. The tf.Variable is
    something similar except it's modifiable and I suppose has advantages over
    placeholders when we need to use weights. BTW for future models we better
    not be using tf.zeros(...) to initialize them.

    For the datasets, the code is currently in:
        tensorflow.contrib.learn.python.learn.datasets.mnist

    The contrib.learn python is supposed to be a non-supported branch designed
    to ease people into TF.
    """
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Now we need to do _training_ so as usual, we have to define a loss
    # function. Use cross-entropy, which goes well with softmax (an alternative
    # loss would be least squares). Use a placeholder for _correct_ answers.
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Note: y_ * tf.log(y) should be the same as numpy's Hadamard product, and
    # the result is a [None,10] matrix with one nonzero in each row, then
    # tf.reduce_sum will turn that into a simple vector by summation.
    #
    # cross_entropy = tf.reduce_mean(
    #     -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    #
    # Use a more numerically stable version of it. (NOTE: It looks like it will
    # be deprecated starting in 2017 ... so why is it still in the tutorial?)
    # BTW I can't find where this method is implemented. It must be in a kernel
    # somewhere in C++ code.  But anyway, then we use the gradient descent
    # optimizer. Yeah, they hide all of it behind the scenes.
    cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # OK, now let's start a "session", then train and test. The next_batch
    # method is Python code in the contrib.learn section of TF, while tf.argmax,
    # tf.equal, and tf.cast are symbolic operations. The argmax is over an axis,
    # of which higher numbers means we go "deeper" into the "numpy" tensor. Note
    # the feed_dict which provides the data. I will be seeing a lot of that! The
    # accuracy is 0.9063.
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # Above command is equivalent to: tf.global_variables_initializer().run()
    # because an Interactivesession() lets us do operation.run().

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='/tmp/tensorflow/mnist/input_data', 
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

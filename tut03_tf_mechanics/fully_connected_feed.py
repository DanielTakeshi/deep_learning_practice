"""
This is based directly off the TF tutorials. -Daniel Seita

Purpose of this code: to train the built MNIST model against the downloaded
dataset using a feed dictionary. A bit confusing, there are two mnist.py scripts
in the repository:

    tensorflow/examples/tutorials/mnist/mnist.py (this is what this code refers
    to for the training)

    tensorflow/contrib/learn/python/learn/datasets/mnist.py (this one is about
    loading the data itself)

I don't need to have any of these here since the first one is imported here and
the second one is implicity loaded via the input_data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
from six.moves import xrange
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    """ Runs one evaluation against the full epoch of data. """

    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        feed_dict = {images_placeholder: images_feed, labels_placeholder: labels_feed}
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % 
            (num_examples, true_count, precision))


def run_training():
    """ 
    Train MNIST for a number of steps. A bit annoying, the
    input_data.read_data_sets is from the tf.contrib.learn library. I'm still
    not sure why they're using unsupported stuff in official tutorials.

    BTW the default tensorflow graph argument is not needed, I think, but this
    is just to be explicit. Then they call a bunch of stuff from mnist but
    that's mostly familiar to me, e.g. name_scope and tf.Variable() stuff. This
    is FC stuff so the shapes are fairly easy to reason about. Once we have the
    graph built, we can train by looping here.
    """
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    with tf.Graph().as_default():
        # The tutorial puts this in another method but it's only two lines. This
        # will get passed into a session by using a feed_dict.
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, mnist.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))

        # inference: Build the MNIST model up to where it may be used for
        # inference.  Builds a FC-net, not a CNN. But the code is familiar to
        # me. The code also splits variable names according to scope: hidden1,
        # hidden2, and softmax_linear, e.g. hidden1/biases and hidden2/biases.
        # 
        # loss: the logits are the final layer output _before_ the softmax. The
        # loss will then do the softmax and compute the cross-entropy loss. It
        # doesn't _have_ to be cross-entropy but it's the one we usually use.
        # 
        # train_op: operations for computing the gradient. Note that this has a
        # summarizer so we can keep track of it in Tensorboard. (I assume that's
        # related with tf.summary.merge_all().)
        #
        # eval_correct: for evaluation. This, along with the previous three, are
        # familiar to me from previous tutorials.
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        loss = mnist.loss(logits, labels_placeholder)
        train_op = mnist.training(loss, FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Some stuff for book-keeping and "administration boilerplate." We need
        # tf.Session() to _create_ the (default) Graph.
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init)

        # OK, now we can do training!
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # The tutorial has this as a separate method but it's fine for me to
            # do it here. The next_batch is coded in Python in mnist.py. This is
            # the same as code I had before. The feed_dict has entries of the
            # form {tf.placeholder : "actual_values"}. The sess.run() method has
            # feed_dict as one of its parameters.
            images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
            feed_dict = {images_placeholder: images_feed, labels_placeholder: labels_feed}

            # In tut01, I did sess.run(train, feed_dict), but here we can have a
            # _list_ as input. What happens is that sess.run will _fetch_ the
            # values in the list. I guess it was doing the same for train but I
            # never needed the output. Here, however, loss_value is a critical
            # parameter for debugging, hence it makes sense to return it. And
            # _fortunately_ the returned values are NUMPY arrays. Yay!
            # Question: how does it know that loss is input to train_op, and
            # that we are really "running" train_op which uses loss as a
            # "subroutine." Does it automatically detect this?
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            # Debugging messages. Huh, even here summary needs a sess.run
            # command. That must be key. Also maybe it's not bad to use
            # sess.run() instead of summary.eval() because then I can clearly
            # see all the sess.run() stuff, and it's not that much extra typing.
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save and evaluate on train/valid/test. It's common to save weights
            # in files so that we can resume training later.
            if ((step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps):
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)


def main(_):
    """ Copied/pasted straight from the tutorial. """
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    """ 
    Lots of settings I can test with! But just copy defaults. All this
    eventually goes into the FLAGS variable.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=20000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

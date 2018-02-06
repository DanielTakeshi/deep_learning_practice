import argparse, random, sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def get_tf_session(gpumem):
    """ Returning a session. Set options here if desired. """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpumem)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


class Evaluator:

    def __init__(self, burn_in_epochs, ylabels):
        """ The input `ylabels` is in one-hot form. """
        self.burn_in_epochs = burn_in_epochs
        self.num_labels = ylabels.shape[0]
        self.rcounter = 0
        self.num_rounds_averaged = 0
        self.o_pred = np.zeros((self.num_labels,10))
        self.y_labels = np.argmax(ylabels, axis=1)
        assert len(self.y_labels.shape) == 1

    def _get_alpha(self):
        if self.rcounter < self.burn_in_epochs:
            return 1.0
        else:
            self.num_rounds_averaged += 1
            return 1.0 / self.num_rounds_averaged

    def eval(self, y_softmax):
        assert y_softmax.shape == self.o_pred.shape
        self.rcounter += 1
        alpha = self._get_alpha()
        self.o_pred[:] *= (1.0 - alpha)
        self.o_pred[:] += alpha * y_softmax
        y_pred = np.argmax(self.o_pred, axis=1)
        num_correct = np.sum( np.equal(y_pred, self.y_labels) )
        return 100 * (1 - (num_correct / float(self.num_labels)))


class Classifier:
    """ Only supports MNIST for now! """

    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        self.mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
        assert self.mnist.train.labels.shape[0] == args.num_train
        assert self.mnist.validation.labels.shape[0] == args.num_valid
        assert self.mnist.test.labels.shape[0] == args.num_test

        # Placeholders and network output.
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.y_logits = self.make_network(self.x)
        self.y_softmax = tf.nn.softmax(self.y_logits)

        # Loss function.
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_logits)
        )
        self.variables = tf.trainable_variables()
        self.l2_loss = args.l2_reg * \
                tf.add_n([ tf.nn.l2_loss(v) for v in self.variables if 'bias' not in v.name ])
        self.loss = self.cross_entropy + self.l2_loss

        # Optimization and statistics.
        self.optimizer = self.get_optimizer()
        self.train_step = self.optimizer.minimize(self.loss)
        self.correct = tf.equal(tf.argmax(self.y_logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
        self.stats = {
            'accuracy': self.accuracy,
            'cross_entropy': self.cross_entropy,
            'l2_loss': self.l2_loss,
            'y_softmax': self.y_softmax,
        }
        self.eval_valid = Evaluator(args.burn_in_epochs, self.mnist.validation.labels)
        self.eval_test  = Evaluator(args.burn_in_epochs, self.mnist.test.labels)
        self.sess.run(tf.global_variables_initializer())
        self.debug()


    def get_optimizer(self):
        args = self.args
        name = (args.optimizer).lower()
        if name == 'sgd':
            return tf.train.GradientDescentOptimizer(args.lrate)
        elif name == 'momsgd':
            return tf.train.MomentumOptimizer(args.lrate, momentum=args.momentum)
        elif name == 'rmsprop':
            return tf.train.RMSPropOptimizer(args.lrate)
        else:
            raise ValueError()


    def make_network(self, x):
        if self.args.net_type == 'ff':
            return self.make_ff(x)
        elif self.args.net_type == 'cnn':
            return self.make_cnn(x)
        else:
            raise ValueError()


    def debug(self):
        print("Here are the variables in our network:")
        for item in self.variables:
            print(item)
        print("(End of debug prints)\n")


    def make_ff(self, x):
        size = self.args.fc_size
        with tf.variable_scope('ff'):
            x = tf.nn.relu( tf.keras.layers.Dense(size)(x) )
            x = tf.nn.relu( tf.keras.layers.Dense(size)(x) )
            x = tf.keras.layers.Dense(10)(x)
            return x


    def make_cnn(self, x):
        x = tf.transpose(tf.reshape(x, [-1, 1, 28, 28]), [0, 2, 3, 1])
        with tf.variable_scope('cnn'):
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='SAME')(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x) # shape = (?, 14, 14, 32)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=[5,5], padding='SAME')(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x) # shape = (?, 7, 7, 64)
            x = tf.keras.layers.Flatten()(x) # shape = (?, 7*7*64) = (?, 3136)
            x = tf.nn.relu( tf.keras.layers.Dense(200)(x) )
            x = tf.nn.relu( tf.keras.layers.Dense(200)(x) )
            x = tf.keras.layers.Dense(10)(x)
            return x


    def train(self):
        """
        By default, `shuffle=True` for `next_batch` but the code internally
        shuffles for the first epoch, then goes through elements sequentially.
        """
        args = self.args
        mnist = self.mnist
        feed_valid = {self.x: mnist.validation.images, self.y: mnist.validation.labels}
        feed_test = {self.x: mnist.test.images, self.y: mnist.test.labels}
        print("epoch | l2_loss (v) | ce_loss (v) | valid_err (s) | valid_err (m) | test_err (s) | test_err (m)")

        for ep in range(args.num_epochs):
            num_mbs = int(args.num_train / args.batch_size)
            for _ in range(num_mbs):
                batch = mnist.train.next_batch(args.batch_size)
                feed = {self.x: batch[0], self.y: batch[1]}
                self.sess.run(self.train_step, feed)
            valid_stats = self.sess.run(self.stats, feed_valid)
            test_stats  = self.sess.run(self.stats, feed_test)

            valid_err_single = 100*(1.0-valid_stats['accuracy'])
            valid_err_model  = self.eval_valid.eval(valid_stats['y_softmax'])
            test_err_single  = 100*(1.0-test_stats['accuracy'])
            test_err_model   = self.eval_test.eval(test_stats['y_softmax'])

            print("{:5} {:9.4f} {:9.4f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}".format(ep,
                    valid_stats['l2_loss'], valid_stats['cross_entropy'],
                    valid_err_single, valid_err_model,
                    test_err_single, test_err_model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Bells and whistles
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data')
    parser.add_argument('--seed', type=int, default=1)
    # Training and evaluation, stuff that should stay mostly constant:
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=200) # just run longer
    parser.add_argument('--momentum', type=float, default=0.99)
    # Training and evaluation, stuff to mostly tune:
    parser.add_argument('--burn_in_epochs', type=int, default=20)
    parser.add_argument('--lrate', type=float, default=0.2)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    # Network and data. the 784-400-400-10 seems a common benchmark.
    parser.add_argument('--fc_size', type=int, default=400)
    parser.add_argument('--net_type', type=str, default='ff')
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--num_train', type=int, default=55000)
    parser.add_argument('--num_valid', type=int, default=5000)
    args = parser.parse_args()
    print("Our arguments:\n{}".format(args))

    sess = get_tf_session(gpumem=1.0)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    classifier = Classifier(args, sess)
    classifier.train()

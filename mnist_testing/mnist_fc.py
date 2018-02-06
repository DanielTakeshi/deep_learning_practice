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


class NNEvaluator:
    """ Based on Tianqi Chen's stuff. """

    def __init__( self, nnet, xdatas, ylabels, param, prefix='' ):
        self.nnet = nnet
        self.xdatas  = xdatas
        self.ylabels = ylabels
        self.param = param
        self.prefix = prefix
        nbatch, nclass = nnet.o_node.shape
        assert xdatas.shape[0] == ylabels.shape[0]
        assert nbatch == xdatas.shape[1]
        assert nbatch == ylabels.shape[1]
        self.o_pred  = np.zeros( ( xdatas.shape[0], nbatch, nclass ), 'float32'  )
        self.rcounter = 0
        self.sum_wsample = 0.0

    def _get_alpha( self ):
        if self.rcounter < self.param.num_burn:
            return 1.0
        else:
            self.sum_wsample += self.param.wsample
            return self.param.wsample / self.sum_wsample
        
    def eval( self, rcounter, fo ):
        self.rcounter = rcounter
        alpha = self._get_alpha()        
        self.o_pred[:] *= (1.0 - alpha)
        sum_bad  = 0.0
        sum_loglike = 0.0
       
        for i in xrange( self.xdatas.shape[0] ):
            self.o_pred[i,:] += alpha * self.nnet.predict( self.xdatas[i] )
            y_pred = np.argmax( self.o_pred[i,:], 1 )            
            sum_bad += np.sum(  y_pred != self.ylabels[i,:] )
            for j in xrange( self.xdatas.shape[1] ):
                sum_loglike += np.log( self.o_pred[ i , j, self.ylabels[i,j] ] )

        ninst = self.ylabels.size
        fo.write( ' %s-err:%f %s-nlik:%f' % ( self.prefix, sum_bad/ninst, self.prefix, -sum_loglike/ninst) )


class Classifier:

    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        self.mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.y_logits = self.make_network(self.x)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_logits)
        )
        self.variables = tf.trainable_variables()
        self.l2_loss = args.l2_reg * \
                tf.add_n([ tf.nn.l2_loss(v) for v in self.variables if 'bias' not in v.name ])
        self.loss = self.cross_entropy + self.l2_loss
        self.optimizer = self.get_optimizer()
        self.train_step = self.optimizer.minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.y_logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())
        self.debug()


    def get_optimizer(self):
        args = self.args
        name = (args.optimizer).lower()
        if name == 'sgd':
            return tf.train.GradientDescentOptimizer(args.lrate)
        elif name == 'momsgd':
            return tf.train.MomentumOptimizer(args.lrate, momentum=0.99)
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
        Note that by default, `shuffle=True` for `next_batch` but this is fine,
        the code internally shuffles for the first epoch but goes through
        elements sequentially, as I'd expect.
        """
        args = self.args
        mnist = self.mnist
        stuff = [self.accuracy, self.cross_entropy, self.l2_loss]
        print("epoch | l2_loss | ce_loss | test_err (single) | test_err (model)")

        for ep in range(args.num_epochs):
            num_mbs = int(60000 / args.batch_size)
            for _ in range(num_mbs):
                batch = mnist.train.next_batch(args.batch_size)
                feed = {self.x: batch[0], self.y: batch[1]}
                self.sess.run(self.train_step, feed)
            feed = {self.x: mnist.test.images, self.y: mnist.test.labels}
            acc, ce_loss, l2_loss = self.sess.run(stuff, feed)
            print("{:5} {:9.4f} {:9.4f} {:10.2f} {:12.2f}".format(
                    ep, l2_loss, ce_loss, 100*(1-acc), 100*(1-acc)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data')
    parser.add_argument('--lrate', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fc_size', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--net_type', type=str, default='ff0')
    parser.add_argument('--optimizer', type=str, default='sgd')
    args = parser.parse_args()

    sess = get_tf_session(gpumem=1.0)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    classifier = Classifier(args, sess)
    classifier.train()

"""Practice with loading."""
import argparse, random, sys, datetime
import tensorflow as tf
import numpy as np
np.set_printoptions(edgeitems=10, suppress=True)
mnist = tf.keras.datasets.mnist
slim = tf.contrib.slim


def debug():
    # Helpful to understand neural network weights.
    variables = tf.trainable_variables()
    print("\ninside debug()")
    for v in variables:
        print(v)
    print("finished debug()\n")

    # Only if you want to print everything. Note: doesn't include `:0` part.
    if False:
        names = sorted([tensor.name 
                for tensor in tf.get_default_graph().as_graph_def().node])
        for nn in names:
            print(nn)
        sys.exit()


def load():
    # load data (should be shuffled already) and build graph
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    for ep in range(0,9):
        # It's a two-step process. For restoring, don't include the stuff
        # from the `data`, i.e. use `name`, not `name.data-00000-of-00001`.
        sess = tf.Session()
        saver = tf.train.import_meta_graph('checkpoints/epoch-{}.meta'.format(ep))
        saver.restore(sess, 'checkpoints/epoch-{}'.format(ep))
        if ep == 0:
            debug()

        # How we extract individual tensors. Note the `:0`.
        graph = tf.get_default_graph()
        images_ph = graph.get_tensor_by_name("images:0")
        labels_ph = graph.get_tensor_by_name("labels:0")
        accuracy_op = graph.get_tensor_by_name("accuracy_op:0")
        cross_entropy_op = graph.get_tensor_by_name("cross_entropy_op:0")

        # Test and see if we get same test-time performance.
        bs = 100
        cum_acc = 0.0
        cum_loss = 0.0
        k = 0
        for start in range(0, 10000, bs):
            k += 1
            xs = np.expand_dims(x_test[start : start+bs], axis=3)
            ys = y_test[start : start+bs]
            feed_dict = {images_ph:xs, labels_ph:ys}
            acc_test, loss_test = sess.run([accuracy_op, cross_entropy_op], feed_dict)
            cum_acc += acc_test
            cum_loss += loss_test
        acc_test = cum_acc / float(k)
        loss_test = cum_loss / float(k)

        print("{}, {:.3f}, {:.5f}".format(ep+1, acc_test, loss_test))


if __name__ == '__main__':
    load()

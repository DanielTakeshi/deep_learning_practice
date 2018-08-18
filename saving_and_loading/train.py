""" Practice with `tf.train.Saver`. """
import argparse, random, sys, datetime
import tensorflow as tf
import numpy as np
np.set_printoptions(edgeitems=10, suppress=True)
mnist = tf.keras.datasets.mnist
slim = tf.contrib.slim


def train_keras():
    """Only here for completeness, to show how easy keras makes it."""
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


def train_slim():
    # load data (should be shuffled already) and build graph
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(x_train.shape, x_train.dtype)
    print(y_train.shape, y_train.dtype)
    print(x_test.shape, x_test.dtype)
    print(y_test.shape, y_test.dtype)

    # For one-hot, depth=10 because there are 10 classes.
    images_ph = tf.placeholder(tf.float32, [None, 28, 28, 1], name='images')
    labels_ph = tf.placeholder(tf.float32, [None], name='labels')
    labels_ph_one_hot = tf.one_hot(tf.cast(labels_ph, tf.int32), depth=10)

    print("\nhere's the network input style:")
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
        net = images_ph
        print(net)
        net = slim.conv2d(net, 16, [5, 5], 1)
        print(net)
        net = slim.max_pool2d(net, [2, 2], 2)
        print(net)
        net = slim.conv2d(net, 16, [3, 3])
        print(net)
        net = slim.max_pool2d(net, [2, 2], 2)
        print(net)
        net = slim.flatten(net)
        print(net)
        net = slim.fully_connected(net, 100)
        print(net)
        net = slim.fully_connected(net, 100)
        print(net)
        net = slim.fully_connected(net, 10, activation_fn=None)
        print(net)
    print("")

    logits_ph = net

    # Training Operations
    # Note: use `_v2` as original version of softmax(C.E.) is deprecated.
    cross_entropy_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_ph_one_hot, logits=logits_ph),
        name='cross_entropy_op'
    )
    correct_op = tf.equal(tf.argmax(logits_ph, 1), tf.argmax(labels_ph_one_hot, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32), name='accuracy_op')
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_op)

    # Initialize. Also, tf.train.Saver() must be done _after_ creating variables
    # First argument is `var_list` but if None, then save all saveable variables
    # Set `max_to_keep=None` since otherwise we'll be deleting earlier checkpoints
    # https://www.tensorflow.org/api_docs/python/tf/train/Saver
    saver = tf.train.Saver(max_to_keep=None)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train
    bs = 100
    print("epoch, test_accuracy, test_loss")

    for ep in range(0,10):
        # I know, lame, only works if even number of testing instances in each batch
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
        print("{}, {:.3f}, {:.5f}".format(ep, acc_test, loss_test))

        # After evaluation b/c I wanted to see performance before any training
        for start in range(0, 60000, bs):
            xs = np.expand_dims(x_train[start : start+bs], axis=3)
            ys = y_train[start : start+bs]
            feed_dict = {images_ph:xs, labels_ph:ys}
            _, loss_train = sess.run([train_op, cross_entropy_op], feed_dict)

        # Can use `curr_time` if you want unique file names.
        # curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        # I save with `epochs` there, via a `global_step`.
        # Will create:
        # epoch-0.data-00000-of-00001 (i.e., the weights!)
        # epoch-0.index (a way to link values with variables)
        # epoch-0.meta (meta-data, etc.)
        # etc....
        # https://www.tensorflow.org/guide/checkpoints
        # https://www.tensorflow.org/guide/saved_model
        ckpt_name = "checkpoints/epoch"
        saver.save(sess, ckpt_name, global_step=ep)


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    train_slim()

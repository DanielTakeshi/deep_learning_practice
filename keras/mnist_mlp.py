"""
My first attempt in learning Keras. This is so much easier than using default
TensorFlow so let me try using this for DQN. WOW, they have a long list of
examples on their GitHub. This is really helpful. I get for instance:

Test score: 0.100619024031
Test accuracy: 0.9846

For more on the actual API documentation, see:
    https://keras.io/layers/core/
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337) 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


if __name__ == "__main__":
    batch_size = 128
    nb_classes = 10
    nb_epoch = 20

    # Yay! These are _direct_ numpy arrays. Very familiar from CS 231n.
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # I get it, convert vectors y_train, y_test into matrices where rows are the
    # one-hot vectors for the particular example. Note, this is not in numpy but
    # a keras-specific utility function.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # Now here's where things get interesting/new to me. First argument for
    # Dense is the number of hidden units in that layer, second is the shape.
    # There's at least three equivalent forms of expressing the shape. I prefer
    # the batch_input_shape since it makes it explicit the input dimensions.
    model = Sequential()
    model.add(Dense(output_dim=512, batch_input_shape=(None,784), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary() # WOW, omg thanks Keras.
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # Did I mention I like Keras? Use verbose=1, it's very handy. And yes we're
    # using the test set as the validation, lol ... sorry, bad practice. The
    # 'fit' function is what we usually use to train a model.
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_data=(X_test,Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

"""
OK let's experiment a bit with their MNIST CNN example. Lots of layers to
understand. See the documentation:

    https://keras.io/layers/

I get:

    Test score: 0.0312790817739
    Test accuracy: 0.9894

Maybe it's better to separate 'Dense' and 'Activation'? It helps when viewing
the model summary. This model here looks like:

(batch_size, 28, 28, 1)
(batch_size, 26, 26, 32) // after first conv, w/32 filters
(batch_size, 24, 24, 32) // after second conv (_no_ max pooling beforehand)
(batch_size, 12, 12, 32) // after max-pooling for first time
(bath_size, 12*12*32) // after flattening

I see, just like in TensorFlow, if you say the output of a conv has 32 channels,
that literally means the channel will be 32 dimensions long.
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# The actual tutorial makes a special case because it has to test for Theano vs
# TensorFlow due to differences in how the two "backends" (that's what we
# imported above) order images. Since I know I'm using TF, I'm avoiding the if
# block. Otherwise, most of this is the same as the MLP example. Note the need
# for 28x28 for CNNs, which we didn't need to use for the MLP.
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1) # for Convolution2D's API
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Done with data loading!! Time for the big guns. The Convolution2D API is again
# pretty simple. The 'stride' isn't here because it's a default parameter,
# actually called 'subsample', but the default is a stride of 1 in all
# dimensions, so we're not skipping potential blocks of 3x3 (that's the kernel
# size from earlier). Also, notice again that the input_shape is only necessary
# for the _first_ layer on the network.
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten()) # I see, the transition from convs to FCs
model.add(Dense(output_dim=128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=nb_classes))
model.add(Activation('softmax'))

# Now back to the usual compilation. Note the ADADelta optimizer. Also, quick
# question, does the loss include weight regularization? If not, I think that
# has to be added in the Convolution2D and Dense layers.
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

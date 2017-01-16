"""
Third Keras testing for me. After this I'm pretty much done the basis of keras
with fully connected and conv-nets for small architectures. I'll test with
fanicer stuff with RNNs, LSTMs, etc. after this. One interesting thing about
this code is the _data_augmentation_, which might be useful for robotics work (I
think I remember some other students talking about augmenting data). 

Note: for some reason it seems like this model is _overfitting_. The validation
performance can get as high as 70% then go crashing down. I also added weight
regularization so maybe that will help.

Update: OK I shrunk the model a bit. I think there were too many parameters in
the version on GitHub. This way works a little better but performance on the
validation set peaks in the low 60s percentages. In CS 231n, Homework 2, we had
to get 65% accuracy, though that was on a smaller dataset. If I had more time, I
would tune this more and perform some of the checks that Andrej Karpathy talked
about during CS 231n.
"""

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2 # note this addition!

batch_size = 32
nb_classes = 10
nb_epoch = 30
data_augmentation = True

# input image dimensions (CIFAR10 are RGB so 3 channels)
img_rows, img_cols = 32, 32
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
in_shape=X_train.shape[1:]

model = Sequential()
reg = 0.0001

# first convolution, with 32 output filters (32 channels).
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=in_shape,
                        W_regularizer=l2(reg), b_regularizer=l2(reg)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, 
                        W_regularizer=l2(reg), b_regularizer=l2(reg)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3,
                        W_regularizer=l2(reg), b_regularizer=l2(reg)))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3, 
#                        W_regularizer=l2(reg), b_regularizer=l2(reg)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, 
              Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test,Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # See the code's comments, lots of cool stuff! I hope they're not augmenting
    # the test set!
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True,
        vertical_flip=False)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))

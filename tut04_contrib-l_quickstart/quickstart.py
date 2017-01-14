"""
My first experience with using tf.contrib.learn. This, along with
tf.contrib.slim, tflearn, and keras, are high-level libraries built on top of TF
which help eliminate boilerplate code. Ah, I see. =) I will try tf.contrib.learn
for now.

This tutorial represents another way to load data. Unfortunately, it's a bit
hard to use this for data in the wild, but w/e. Fortunately, the 3-layer DNN
code is much simpler and smaller than default TF. Notice that the classifier has
built-in fit, evaluate, and predict methods. And none of this requires lots of
TF session tracking, etc. Definitely useful!

I do indeed get the 0.96667 accuracy. Unfortunately, I get a ton of warnings.
Maybe it's best if I _don't_ use tf.contrib.learn? Maybe keras is more stable.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Now build a 3-layer DNN with 10-20-10 units. Then 'fit' it. Basically a
# two-liner!!! =) For the documentation in tf.contrib.learn, see:
# https://www.tensorflow.org/api_docs/python/contrib.learn/estimators#DNNClassifier
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir='/tmp/iris_model')
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# Evaluate performance, then try new samples using classifier.predict.
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)['accuracy']
print('Accuracy: {0:f}'.format(accuracy_score))
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))

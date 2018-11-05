#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This showcases how simple it is to build image classification networks.

It follows description from this TensorFlow tutorial:
    https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import re
import numpy as np
import tensorflow as tf
import argparse

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

DNN_TF_CPP_ROOT = _d(_d(_a(__file__)))

N_DIGITS = 10  # Number of digits.
X_FEATURE = 'x'  # Name of the input feature.


def conv_model(features, labels, mode):
  """2-layer convolution model."""
  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(features[X_FEATURE], [-1, 28, 28, 1])

  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = tf.layers.conv2d(
      feature,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu)
    h_pool1 = tf.layers.max_pooling2d(
      h_conv1, pool_size=2, strides=2, padding='same')

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = tf.layers.conv2d(
      h_pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu)
    h_pool2 = tf.layers.max_pooling2d(
      h_conv2, pool_size=2, strides=2, padding='same')
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Densely connected layer with 1024 neurons.
  h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)
  h_fc1 = tf.layers.dropout(
    h_fc1,
    rate=0.5,
    training=(mode == tf.estimator.ModeKeys.TRAIN))

  # Compute logits (1 per class) and compute loss.
  logits = tf.layers.dense(h_fc1, N_DIGITS, activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'class': predicted_classes,
      'prob': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Create training op.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  # Compute evaluation metrics.
  eval_metric_ops = {
    'accuracy': tf.metrics.accuracy(
      labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
    mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_args):
  tf.logging.set_verbosity(tf.logging.INFO)

  ### Download and load MNIST dataset.
  mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={X_FEATURE: mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={X_FEATURE: mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=1,
    shuffle=False)

  ### Linear classifier.
  feature_columns = [
    tf.feature_column.numeric_column(
      X_FEATURE, shape=mnist.train.images.shape[1:])]

  classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns, n_classes=N_DIGITS)
  classifier.train(input_fn=train_input_fn, steps=200)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (LinearClassifier): {0:f}'.format(scores['accuracy']))

  ### Convolutional network
  classifier = tf.estimator.Estimator(model_fn=conv_model)
  classifier.train(input_fn=train_input_fn, steps=200)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (conv_model): {0:f}'.format(scores['accuracy']))

def line_iter(path):
  with open(path) as f:
    for line in f:
      line = line.rstrip()
      yield line

class CarExample:
  def __init__(self, parser, args):
    self.parser = parser
    self.args = args

    self.header, self.df = self.read_csv()
    self.batch_size = len(self.df)
    self.sess = tf.InteractiveSession()
    self.build_car_model()

  def run(self):
    self.training_loop()
    self.inference()

  def read_csv(self):
    args = self.args
    it = line_iter(args.csv)

    split_re = r','

    def read_floats(line):
      return [float(x) for x in re.split(split_re, line)]

    def read_strings(line):
      return re.split(split_re, line)

    # data_set_metadata.mean_km = metadata[0];
    # data_set_metadata.std_km = metadata[1];
    # data_set_metadata.mean_age = metadata[2];
    # data_set_metadata.std_age = metadata[3];
    # data_set_metadata.min_price = metadata[4];
    # data_set_metadata.max_price = metadata[5];
    md_line = read_floats(next(it))
    self.md = {
      "mean_km":md_line[0],
      "std_km":md_line[1],
      "mean_age":md_line[2],
      "std_age":md_line[3],
      "min_price":md_line[4],
      "max_price":md_line[5],
    }

    header = read_strings(next(it))
    data = dict()
    for line in it:
      values = read_floats(line)
      for field, value in zip(header, values):
        if field not in data:
          data[field] = []
        data[field].append(value)
    df = pd.DataFrame(data)
    return header, df

  def as_price(self, prediction):
    return prediction * (self.md['max_price'] - self.md['min_price']) + self.md['min_price']

  @property
  def features(self):
    return self.header[:-1]

  @property
  def label(self):
    return self.header[-1]

  def build_car_model(self):
    args = self.args

    self.x = tf.placeholder(tf.float32, shape=(None, len(self.features)), name='features')
    self.y = tf.placeholder(tf.float32, shape=(None,), name='labels')

    # self.x = tf.placeholder(tf.float32, None)
    # self.y = tf.placeholder(tf.float32, None)

    # import ipdb; ipdb.set_trace()
    net = self.x
    act = tf.nn.tanh
    reg = tf.contrib.layers.l2_regularizer(scale=args.regularization_constant)
    net = tf.layers.dense(net, 3, activation=act, kernel_regularizer=reg)
    net = tf.layers.dense(net, 2, activation=act, kernel_regularizer=reg)
    net = tf.layers.dense(net, 1, activation=act, kernel_regularizer=reg)
    self.predictions = net
    # mse_loss = tf.losses.mean_squared_error(self.y[:,np.newaxis], self.predictions)
    mse_loss = tf.losses.mean_squared_error(self.y, tf.squeeze(self.predictions))
    # mse_loss = tf.reduce_mean(tf.squeeze(predictions - labels))
    reg_loss = tf.losses.get_regularization_loss()
    optim = tf.train.GradientDescentOptimizer(args.lr)
    self.loss = mse_loss + reg_loss
    self.step = optim.minimize(self.loss)

  def training_loop(self):
    args = self.args
    self.sess.run(tf.global_variables_initializer())
    x_data = self.df[self.features].values
    y_data = self.df[self.label].values
    for i in range(args.training_steps):
      if i % 100 == 0:
        loss = self.sess.run(self.loss, {self.x:x_data, self.y:y_data})
        print("Loss after {i} steps = {loss}".format(i=i, loss=loss))
      self.sess.run(self.step, {self.x:x_data, self.y:y_data})

  def inference(self):
    FUEL_DIESEL = 0
    FUEL_GASOLINE = 1
    xs = np.array([110000., FUEL_DIESEL, 7.])
    xs = xs.reshape((1, len(xs)))
    predictions = self.sess.run(self.predictions, {self.x:xs})
    predicted_price = self.as_price(np.squeeze(predictions))
    print("Price prediction = {pred} euros".format(pred=predicted_price))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', default=_j(DNN_TF_CPP_ROOT, 'normalized_car_features.csv'))
  parser.add_argument('--training-steps', type=int, default=5000)
  parser.add_argument('--regularization-constant', type=int, default=0.01)
  parser.add_argument('--lr', type=int, default=0.01)
  args = parser.parse_args()

  car_example = CarExample(parser, args)

  car_example.run()

  # tf.app.run()

if __name__ == '__main__':
  main()

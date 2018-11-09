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

import sys
import pandas as pd
import re
import numpy as np
import argparse
import os
import shutil
import pprint

from tensorflow import pywrap_tensorflow as pywrap

import tensorflow as tf
# e.g.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/python/graph_matcher_test.py
from tensorflow.contrib.quantize.python import graph_matcher

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

DNN_TF_CPP_ROOT = _d(_d(_a(__file__)))

MODEL_PATH = _j(DNN_TF_CPP_ROOT, "checkpoints", "model", "model_checkpoint")

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

FUEL_DIESEL = 0
FUEL_GASOLINE = 1

class CarExample:
  def __init__(self, parser, args):
    self.parser = parser
    self.args = args

    self.expected_attrs = [
      # Placeholder.
      'features',
      'labels',
      # NN output.
      'predictions',
      # What we minimize via SGD.
      'loss',
      # Single step of SGD.
      'step',
    ]

  def _check_model_loaded(self):
    for attr in self.expected_attrs:
      assert hasattr(self, attr)

  def read_data(self):
    self.header, self.df = self.read_csv()
    self.batch_size = len(self.df)

  @property
  def model_exists(self):
    return _e(_j(self.args.model_path, 'saved_model.pbtxt')) or \
           _e(_j(self.args.model_path, 'saved_model.pb'))

  def run(self):
    self.read_data()

    self.sess = tf.InteractiveSession()

    if self.args.rebuild_model and _e(self.args.model_path):
      # print("> rm({path})".format(path=self.args.model_path))
      shutil.rmtree(self.args.model_path)
      # sys.exit(1)

    if not self.model_exists:
      self.define_model()
      self.training_loop()
      self.save_model()
    else:
      self.load_model()
    self._check_model_loaded()

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
    print("> Metadata")
    print("    mean_km = {}".format(self.md['mean_km']))
    print("    std_km = {}".format(self.md['std_km']))
    print("    mean_age = {}".format(self.md['mean_age']))
    print("    std_age = {}".format(self.md['std_age']))
    print("    min_price = {}".format(self.md['min_price']))
    print("    max_price = {}".format(self.md['max_price']))

    # Python:
    # > Metadata
    # mean_km = 104272.93201133145
    # std_km = 65391.70758258462
    # mean_age = 6.016883852691218
    # std_age = 3.40778120276032
    # min_price = 1500.0
    # max_price = 124000.0

    # C++: difference is b/c C-float vs python-double?
    # Metadata:
    # mean_km:104273
    # std_km:65391.7
    # mean_age:6.01688
    # std_age:3.40778
    # min_price:1500
    # max_price:124000

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

  def normalize_input(self, km, fuel, age):
    pass
    km = (km - self.md['mean_km']) / self.md['std_km'];
    age = (age - self.md['mean_age']) / self.md['std_age'];
    if fuel == FUEL_DIESEL:
      f = -1.
    else:
      f = 1.
    return np.array([km, f, age])

  @property
  def feature_names(self):
    return self.header[:-1]

  @property
  def label(self):
    return self.header[-1]

  def lookup_ops(self, scope=None, sub_scope=None):
    '''
    Lookup operation beloning to a given scope.
    '''
    assert scope is not None or sub_scope is not None

    if scope is not None:
      scope_regex = r'^{scope}/'.format(scope=scope)
    else:
      scope_regex = r'\b{scope}/'.format(scope=sub_scope)

    return [op for op in tf.get_default_graph().get_operations() \
            if re.search(scope_regex, op.name)]

  def lookup_predictions(self, scope):
    """
    ipdb> self.predictions
    <tf.Tensor 'outputs/predictions/Tanh:0' shape=(?, 1) dtype=float32>
    """
    print("> Lookup predictions")

    # ops = self.lookup_ops(scope=scope)
    # pred_op = ops[-1]
    # tensor = self._tensor_from_ops([pred_op])
    # # Not clear to me if we're guaranteed that pred_op will be the last operation...
    # assert tensor.name == 'outputs/predictions/Tanh:0'
    # return tensor

    # More direct method:
    tensor = tf.get_default_graph().get_tensor_by_name('outputs/predictions/Tanh:0')
    return tensor

  def lookup_loss(self, scope):
    """
    ipdb> g.get_collection('losses')
      [<tf.Tensor 'mean_squared_error/value:0' shape=() dtype=float32>]
    ipdb> pp g.get_collection('regularization_losses')
      [<tf.Tensor 'dense/kernel/Regularizer/l2_regularizer:0' shape=() dtype=float32>,
      <tf.Tensor 'dense_1/kernel/Regularizer/l2_regularizer:0' shape=() dtype=float32>,
      <tf.Tensor 'outputs/predictions/kernel/Regularizer/l2_regularizer:0' shape=() dtype=float32>]

    PROBLEM: What we really want to "get" here is the input to optimizer.minimize(...).

    ipdb> g.get_tensor_by_name('loss/loss:0')
    <tf.Tensor 'loss/loss:0' shape=() dtype=float32>
    """
    print("> Lookup loss")

    ops = self.lookup_ops(scope=scope)
    # ipdb> self.lookup_ops(scope='loss')
    # [<tf.Operation 'loss/loss' type=Add>]

    tensor = self._tensor_from_ops(ops)
    assert tensor.name == 'loss/loss:0'

    return tensor

    # return tf.get_default_graph().get_tensor_by_name('loss/loss:0')

    # More direct method:
    # g.get_operation_by_name('loss/loss')

  def _tensor_from_ops(self, ops):
    assert len(ops) == 1
    op = ops[0]
    values = op.values()
    assert len(values) == 1
    value = values[0]
    tensor = value
    return tensor

  def lookup_step(self):
    print("> Lookup step")
    # # ['variables', 'update_ops', 'train_op', 'regularization_losses', 'cond_context', 'losses', 'trainable_variables']
    # assert tf.GraphKeys.TRAIN_OP in tf.get_default_graph().get_all_collection_keys()
    # coll = tf.get_default_graph().get_collection(tf.GraphKeys.TRAIN_OP)
    # assert len(coll) == 1
    # return coll[0]

    # More direct method:
    tf.get_default_graph().get_operation_by_name('step/step')

    # > Trainable variables
    # { "<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32_ref>": array([-0.10614621,  0.01781176, -0.03666358], dtype=float32),
    #   "<tf.Variable 'dense/kernel:0' shape=(3, 3) dtype=float32_ref>": array([[-0.04800478, -0.5441852 ,  0.36759377],
    #                                                                           [ 0.4027491 ,  0.0943684 ,  0.15645161],
    #                                                                           [-0.42262703,  0.05545909, -0.48662126]], dtype=float32),
    #   "<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32_ref>": array([0.01540005, 0.14128086], dtype=float32),
    #   "<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32_ref>": array([[-0.53406185, -0.55536765],
    #                                                                             [-0.27319422,  0.22930974],
    #                                                                             [ 0.12702847, -0.00354132]], dtype=float32),
    #   "<tf.Variable 'predictions/bias:0' shape=(1,) dtype=float32_ref>": array([0.14161532], dtype=float32),
    #   "<tf.Variable 'predictions/kernel:0' shape=(2, 1) dtype=float32_ref>": array([[-0.18554956],
    #                                                                                 [ 0.00688495]], dtype=float32)}

    # Variable dense/bias/read = [-0.106146, 0.0178118, -0.0366636]
    # Variable dense/kernel/read = [-0.0480048, -0.544185, 0.367594, 0.402749, 0.0943684, 0.156452, -0.422627, 0.0554591, -0.486621]
    # Variable dense_1/bias/read = [0.0154001, 0.141281]
    # Variable dense_1/kernel/read = [-0.534062, -0.555368, -0.273194, 0.22931, 0.127028, -0.00354132]
    # Variable predictions/bias/read = [0.141615]
    # Variable predictions/kernel/read = [-0.18555, 0.00688495]

  def print_ops(self):
    ops = tf.get_default_graph().get_operations()
    print("> Graph operations")
    pprint.pprint(ops, indent=2)

  def print_vars(self):
    variables = tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print("> Trainable variables")
    var_dict = dict((str(v), v.eval()) for v in variables)
    pprint.pprint(var_dict, indent=2)

  def define_model(self):
    print("> Define model")
    args = self.args

    # with tf.variable_scope('model'):

    with tf.name_scope('inputs'):
      self.features = tf.placeholder(tf.float32, shape=(None, len(self.feature_names)), name='features')
      self.labels = tf.placeholder(tf.float32, shape=(None,), name='labels')

    net = self.features
    act = tf.nn.tanh
    reg = tf.contrib.layers.l2_regularizer(scale=args.regularization_constant)
    net = tf.layers.dense(net, 3, activation=act, kernel_regularizer=reg)
    net = tf.layers.dense(net, 2, activation=act, kernel_regularizer=reg)
    with tf.name_scope('outputs'):
      net = tf.layers.dense(net, 1, activation=act, kernel_regularizer=reg, name='predictions')
      # <tf.Tensor 'predictions/Tanh:0' shape=(?, 1) dtype=float32>
      self.predictions = net
    # mse_loss = tf.losses.mean_squared_error(self.labels[:,np.newaxis], self.predictions)
    mse_loss = tf.losses.mean_squared_error(self.labels, tf.squeeze(self.predictions))
    # mse_loss = tf.reduce_mean(tf.squeeze(predictions - labels))
    reg_loss = tf.losses.get_regularization_loss()
    optim = tf.train.GradientDescentOptimizer(args.lr)
    with tf.name_scope('loss'):
      self.loss = tf.add(mse_loss, reg_loss, name='loss')
    with tf.name_scope('step'):
      self.step = optim.minimize(self.loss, name='step')

    print("> Model defined.")

  def training_loop(self):
    print("> Training loop")
    args = self.args
    self.sess.run(tf.global_variables_initializer())
    x_data = self.df[self.feature_names].values
    y_data = self.df[self.label].values
    for i in range(args.training_steps):
      if i % 100 == 0:
        loss = self.sess.run(self.loss, {self.features:x_data, self.labels:y_data})
        print("Loss after {i} steps = {loss}".format(i=i, loss=loss))
      self.sess.run(self.step, {self.features:x_data, self.labels:y_data})

  def inference(self):
    print("> Inference")
    xs = self.normalize_input(km=110000., fuel=FUEL_DIESEL, age=7.)
    xs = xs.reshape((1, len(xs)))

    # C++:
    #
    # > Features tensor:
    # array([[ 0.08758095, -1.        ,  0.28849157]])
    #
    # Features tensor: [1.31437e-38, 0, 1.68044e+22]
    # Output value from neural-network: 0.118596
    # Predicted price: 16028
    #
    # Features tensor: [1.31437e-38, 0, -2.82041e-34]
    # Output value from neural-network: 0.130695
    # Predicted price: 17510.2

    print("> Features tensor:")
    pprint.pprint(xs)

    predictions = self.sess.run(self.predictions, {self.features:xs})
    output = np.squeeze(predictions)
    predicted_price = self.as_price(output)
    print("Output value from neural-network = {out}".format(out=output))
    print("Price prediction = {pred} euros".format(pred=predicted_price))

  def save_model(self):
    # Create a builder
    print("> Save model to {path}".format(path=self.args.model_path))
    # os.makedirs(self.args.model_path, exist_ok=True)
    if os.path.isdir(self.args.model_path) and is_empty_dir(self.args.model_path):
      os.rmdir(self.args.model_path)
    builder = tf.saved_model.builder.SavedModelBuilder(self.args.model_path)
    builder.add_meta_graph_and_variables(self.sess,
                                         [tf.saved_model.tag_constants.TRAINING],
                                         signature_def_map=None,
                                         assets_collection=None)
    builder.save()

  def load_model(self):
    print("> Load trained model from {path}".format(path=self.args.model_path))
    tf.saved_model.loader.load(self.sess,
                               [tf.saved_model.tag_constants.TRAINING],
                               self.args.model_path)

    # Sets:
    # self.features and self.labels.
    phs = self.get_placeholders()
    for ph in phs:
      # Remove scope information
      placeholder_name = _b(ph.name)
      ph_tensor = self._tensor_from_ops([ph])
      setattr(self, placeholder_name, ph_tensor)
    # More direct method:
    # g.get_tensor_by_name('inputs/features:0')
    # g.get_tensor_by_name('inputs/labels:0')

    self.step = self.lookup_step()
    self.loss = self.lookup_loss('loss')
    self.predictions = self.lookup_predictions('outputs')

    self.print_ops()
    self.print_vars()

  def get_variables(self):
    return tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES)

  def get_placeholders(self):
    return [op for op in tf.get_default_graph().get_operations() \
            if op.type == 'Placeholder']

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', default=_j(DNN_TF_CPP_ROOT, 'normalized_car_features.csv'))
  parser.add_argument('--training-steps', type=int, default=5000)
  parser.add_argument('--regularization-constant', type=int, default=0.01)
  parser.add_argument('--lr', type=int, default=0.01)
  parser.add_argument('--model-path', default=MODEL_PATH)
  parser.add_argument('--rebuild-model', action='store_true')
  parser.add_argument('--trace-tf', action='store_true', help="Print C-api calls made by python")

  args = parser.parse_args()

  if args.trace_tf:
    pywrap._wrap_tf_functions(debug_print=True)

  car_example = CarExample(parser, args)

  car_example.run()

  # tf.app.run()

def is_empty_dir(path):
  return len(os.listdir(path)) == 0

if __name__ == '__main__':
  main()

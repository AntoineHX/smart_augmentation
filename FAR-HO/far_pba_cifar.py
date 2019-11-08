#https://github.com/arcelien/pba/blob/master/autoaugment/train_cifar.py
from __future__ import absolute_import, print_function, division

import os
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.layers as layers
import far_ho as far
import far_ho.examples as far_ex
#import pprint

import autoaugment.augmentation_transforms as augmentation_transforms
#import autoaugment.policies as found_policies
from autoaugment.wrn import build_wrn_model


def build_model(inputs, num_classes, is_training, hparams):
  """Constructs the vision model being trained/evaled.
  Args:
    inputs: input features/images being fed to the image model build built.
    num_classes: number of output classes being predicted.
    is_training: is the model training or not.
    hparams: additional hyperparameters associated with the image model.
  Returns:
    The logits of the image model.
  """
  scopes = setup_arg_scopes(is_training)
  with contextlib.nested(*scopes):
    if hparams.model_name == 'pyramid_net':
      logits = build_shake_drop_model(
          inputs, num_classes, is_training)
    elif hparams.model_name == 'wrn':
      logits = build_wrn_model(
          inputs, num_classes, hparams.wrn_size)
    elif hparams.model_name == 'shake_shake':
      logits = build_shake_shake_model(
          inputs, num_classes, hparams, is_training)
  return logits


class CifarModel(object):
  """Builds an image model for Cifar10/Cifar100."""

  def __init__(self, hparams):
    self.hparams = hparams

  def build(self, mode):
    """Construct the cifar model."""
    assert mode in ['train', 'eval']
    self.mode = mode
    self._setup_misc(mode)
    self._setup_images_and_labels()
    self._build_graph(self.images, self.labels, mode)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _setup_misc(self, mode):
    """Sets up miscellaneous in the cifar model constructor."""
    self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
    self.reuse = None if (mode == 'train') else True
    self.batch_size = self.hparams.batch_size
    if mode == 'eval':
      self.batch_size = 25

  def _setup_images_and_labels(self):
    """Sets up image and label placeholders for the cifar model."""
    if FLAGS.dataset == 'cifar10':
      self.num_classes = 10
    else:
      self.num_classes = 100
    self.images = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 3])
    self.labels = tf.placeholder(tf.float32,
                                 [self.batch_size, self.num_classes])

  def assign_epoch(self, session, epoch_value):
    session.run(self._epoch_update, feed_dict={self._new_epoch: epoch_value})

  def _build_graph(self, images, labels, mode):
    """Constructs the TF graph for the cifar model.
    Args:
      images: A 4-D image Tensor
      labels: A 2-D labels Tensor.
      mode: string indicating training mode ( e.g., 'train', 'valid', 'test').
    """
    is_training = 'train' in mode
    if is_training:
      self.global_step = tf.train.get_or_create_global_step()

    logits = build_model(
        images,
        self.num_classes,
        is_training,
        self.hparams)
    self.predictions, self.cost = helper_utils.setup_loss(
        logits, labels)
    self.accuracy, self.eval_op = tf.metrics.accuracy(
        tf.argmax(labels, 1), tf.argmax(self.predictions, 1))
    self._calc_num_trainable_params()

    # Adds L2 weight decay to the cost
    self.cost = helper_utils.decay_weights(self.cost,
                                           self.hparams.weight_decay_rate)
 #### Attention: differe implem originale

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())


########################################################

######## PBA ############

#Parallele Cifar model trainer
tf.flags.DEFINE_string('model_name', 'wrn',
                       'wrn, shake_shake_32, shake_shake_96, shake_shake_112, '
                       'pyramid_net')
tf.flags.DEFINE_string('checkpoint_dir', '/tmp/training', 'Training Directory.')
tf.flags.DEFINE_string('data_path', '/tmp/data',
                       'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_integer('use_cpu', 1, '1 if use CPU, else GPU.')
## ???

FLAGS = tf.flags.FLAGS
FLAGS.dataset
FLAGS.data_path
FLAGS.model_name = 'wrn'

hparams = tf.contrib.training.HParams(
      train_size=50000,
      validation_size=0,
      eval_test=1,
      dataset=FLAGS.dataset,
      data_path=FLAGS.data_path,
      batch_size=128,
      gradient_clipping_by_global_norm=5.0)
  if FLAGS.model_name == 'wrn':
    hparams.add_hparam('model_name', 'wrn')
    hparams.add_hparam('num_epochs', 200)
    hparams.add_hparam('wrn_size', 160)
    hparams.add_hparam('lr', 0.1)
    hparams.add_hparam('weight_decay_rate', 5e-4)

data_loader = data_utils.DataSet(hparams)
data_loader.reset()

with tf.Graph().as_default(): #, tf.device('/cpu:0' if FLAGS.use_cpu else '/gpu:0'):
"""Builds the image models for train and eval."""
    # Determine if we should build the train and eval model. When using
    # distributed training we only want to build one or the other and not both.
    with tf.variable_scope('model', use_resource=False):
      m = CifarModel(self.hparams)
      m.build('train')
      #self._num_trainable_params = m.num_trainable_params
      #self._saver = m.saver
    #with tf.variable_scope('model', reuse=True, use_resource=False):
    #  meval = CifarModel(self.hparams)
    #  meval.build('eval')


##### FAR-HO ####
for _ in range(n_hyper_iterations):



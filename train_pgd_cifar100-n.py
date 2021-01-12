"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from torchvision import transforms, datasets, models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision
from torch.utils.data import DataLoader

from pgd.cifar100.model import Model
from pgd.cifar100.pgd_attack import LinfPGDAttack

from PIL import Image
with open('pgd/cifar100/config_cifar100.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
data_root='sar_data/cifar100'
img_size = [28,28,1]
trained_model_path = 'trained_models/sargan_cifar100'
data_root='sar_data/cifar100'
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
def get_data(train_batch_size):
    
    mnist = datasets.CIFAR100(root=data_root, train=True, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True)#.data.float()
    data_transform = Compose([Resize((img_size[0], img_size[1])),ToTensor()])
    
    train_loader = DataLoader(datasets.CIFAR100(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    
    return train_loader
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir-n']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('pgd/cifar100/config_cifar100.json', model_dir)


with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  #saver.restore(sess,tf.train.latest_checkpoint(model_dir))
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0
  train_loader= get_data(batch_size)
  trainiter = iter(cycle(train_loader))
  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch2, y_batch = next(trainiter)
    y_batch=np.array(y_batch,dtype='uint8')
    x_batch2 = np.array(x_batch2.data.numpy().transpose(0,2,3,1))*255
    x_batch=np.zeros([batch_size,img_size[0]*img_size[1]])
    for i in range(batch_size):
        nextimage=Image.fromarray((x_batch2[i]).astype(np.uint8))
        nextimage=nextimage.convert('L')
        x_batch[i]=np.array(nextimage,dtype='float32').reshape([img_size[0]*img_size[1]])/255

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = x_batch#attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    #if ii % num_summary_steps == 0:
      #summary = sess.run(merged_summaries, feed_dict=adv_dict)
      #summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start

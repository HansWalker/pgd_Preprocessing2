"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack


from sargan_models import SARGAN
from sar_utilities import add_gaussian_noise, preprocess_test
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch
tf.reset_default_graph()

def blur(batch):
    img_size = (224,224,1)
    trained_model_path = 'trained_models/sargan_mnist_gaussian_corrupted'
    BATCH_SIZE = 64
    NOISE_STD_RANGE = [0.1, 0.3]
    g2= tf.Graph()
    with g2.as_default():
        #sargan model
        #reshaping the images to a square
        newbatch=batch.reshape([len(batch),28,28])
        #upping the size of the image
        transform=Resize([224,224])
        #creating the original batch
        transformed_batch=np.array(transform(torch.from_numpy(newbatch))).reshape([len(batch),224,224,1])
        #corrupting
        corruptedbatch=np.zeros([len(transformed_batch),224,224,1])
        for i in range(len(transformed_batch)):
            corruptedbatch[i]=gaussian_filter(np.array([add_gaussian_noise(transformed_batch[i], sd=np.random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1]))]), sigma=2)[0]
        #storage spaces for the prcoessed images
        cbatch=np.zeros([BATCH_SIZE,224,224,1])
        outbatch=np.zeros([len(batch),224,224])
        j=0
        k=0
        sargan_model=SARGAN(img_size, BATCH_SIZE, img_channel=1)
        with tf.Session() as sess:
            sargan_saver= tf.train.Saver()    
            sargan_saver = tf.train.import_meta_graph(trained_model_path+'/sargan_mnist.meta');
            sargan_saver.restore(sess,tf.train.latest_checkpoint(trained_model_path));
            for i in range(len(batch)):
                j+=1
                cbatch[i%BATCH_SIZE]=corruptedbatch[i]
                if((j%BATCH_SIZE==0) or ((i+1)%len(batch)==0)):
                    processed_batch=sess.run(sargan_model.gen_img,feed_dict={sargan_model.image: cbatch, sargan_model.cond: cbatch})
                    for l in range(j):
                        outbatch[k]=processed_batch[l,:,:,0]
                        k+=1
                    j=0
        transform2=Resize([28,28])
        outbatch=np.array(transform2(torch.from_numpy(outbatch)))
        outbatch=outbatch.reshape([len(batch),784])
    return outbatch
# Global constants
with open('pgd/config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']
# Set upd the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model()
    attack = LinfPGDAttack(model, 
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
else:
  model = Model()
  attack = LinfPGDAttack(model, 
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      x_batch_adv =blur(x_batch_adv)
      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

      cur_corr_nat, cur_xent_nat = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_adv)

      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))

# Infinite eval loop
while True:
  cur_checkpoint = tf.train.latest_checkpoint(model_dir)

  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                          datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    evaluate_checkpoint(cur_checkpoint)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      print('Waiting for the next checkpoint ...   ({})   '.format(
            datetime.now()),
            end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)


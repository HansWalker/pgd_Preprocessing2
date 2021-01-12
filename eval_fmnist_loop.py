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

from pgd.fmnist.model import Model
from pgd.fmnist.pgd_attack import LinfPGDAttack
from torchvision import transforms, datasets, models
import torchvision
from torch.utils.data import DataLoader

from sargan_dep.sargan_models import SARGAN
from sargan_dep.sar_utilities import add_gaussian_noise, preprocess_test
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch
data_root='sar_data/FMNIST'

def blur(batch):
    #sargan model
    #reshaping the images to a square
    newbatch=batch.reshape([len(batch),img_size[0],img_size[1],img_size[2]])
    #upping the size of the image#corrupting
    corruptedbatch=np.zeros([len(newbatch),img_size[0],img_size[1],img_size[2]])
    for i in range(len(newbatch)):
        corruptedbatch[i]=gaussian_filter(np.array([add_gaussian_noise(newbatch[i], sd=np.random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1]))]), sigma=2)[0]
    return corruptedbatch

def get_data(train_batch_size):
    
    mnist = datasets.FashionMNIST(root=data_root, train=True, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True).data.float()
    
    data_transform = Compose([ToTensor()])#, Normalize((mnist.mean()/255,), (mnist.std()/255,))])
    
    train_loader = DataLoader(datasets.FashionMNIST(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    return train_loader
global_step = tf.contrib.framework
# Global constants
with open('pgd/fmnist/config_fmnist.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir-n']
# Set upd the data, hyperparameters, and the model


img_size = [28,28,1]
img_size2 = (28,28)
trained_model_path = 'trained_models/sargan_fmnist'
BATCH_SIZE = 64
NOISE_STD_RANGE = [0.1, 0.3]

attacks=[]
epsilon=0
model = Model()
for i in range(int(math.ceil(num_eval_examples / eval_batch_size))):
    if eval_on_cpu:
      with tf.device("/cpu:0"):
          
          attack = LinfPGDAttack(model, 
                               epsilon,
                               config['k'],
                               config['a'],
                               config['random_start'],
                               config['loss_func'])
          attacks.append(attack)
    else:
        
        attack = LinfPGDAttack(model, 
                             epsilon,
                             config['k'],
                             config['a'],
                             config['random_start'],
                             config['loss_func'])
        attacks.append(attack)
    epsilon+=config['epsilon']/int(math.ceil(num_eval_examples / eval_batch_size))
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
#summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
    #sys.stdout = open(os.devnull, 'w')
    for i in range(30):
        g2 =tf.Graph()
        g3=tf.Graph()
        epsilonc=0
        fmnist_loop_list=np.zeros([2,int(math.ceil(num_eval_examples / eval_batch_size))])
        fmnist_loop_list_corr=np.zeros([2,int(math.ceil(num_eval_examples / eval_batch_size))])
        with tf.Session() as sess:
        # Restore the checkpoint
            model1 = Model()
            saver.restore(sess, filename);
    #           with tf.device("/cpu:0"):
            # Iterate over the samples batch-by-batch
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
            total_xent_nat = 0.
            total_xent_adv = 0.
            total_corr_nat = 0
            total_corr_adv = 0        
            x_batch_list=[]
            y_batch_list=[]
            x_adv_list=[]
            for ibatch in range(num_batches):
                train_loader= get_data(BATCH_SIZE)
                trainiter = iter(train_loader)
                
                x_batch, y_batch = next(trainiter)
                x_batch=np.array(x_batch).reshape([BATCH_SIZE,img_size[0]*img_size[1]])
                y_batch_list.append(y_batch)
                
                    
                x_batch_adv = attacks[ibatch].perturb(x_batch, y_batch, sess)
                x_batch_list.append(x_batch_adv)
                corruptedbatch =blur(x_batch_adv)
                x_adv_list.append(corruptedbatch)
              
              
        with g2.as_default():
            with tf.Session() as sess2:
                sargan_model=SARGAN(img_size, BATCH_SIZE, img_channel=1)
                sargan_saver= tf.train.Saver()    
                sargan_saver = tf.train.import_meta_graph(trained_model_path+'/sargan_mnist.meta');
                sargan_saver.restore(sess2,tf.train.latest_checkpoint(trained_model_path));
                for ibatch in range(num_batches):
                    processed_batch=sess2.run(sargan_model.gen_img,feed_dict={sargan_model.image: x_adv_list[ibatch], sargan_model.cond: x_adv_list[ibatch]})
                    processed_batch=np.array(processed_batch).reshape([len(processed_batch),img_size[0]*img_size[1]])
                    x_adv_list[ibatch]=processed_batch
        with g3.as_default():
            model3 = Model()
            saver2 = tf.train.Saver()
            with tf.Session() as sess3:
                saver2.restore(sess3, filename);
                for ibatch in range(num_batches):
                    dict_nat = {model3.x_input: x_batch_list[ibatch],
                            model3.y_input: y_batch_list[ibatch]}
                        
                    dict_adv = {model3.x_input: x_adv_list[ibatch],
                                model3.y_input: y_batch_list[ibatch]}
                
                    cur_corr_nat, cur_xent_nat = sess3.run(
                        [model3.num_correct,model3.xent],
                        feed_dict = dict_nat)
                    cur_corr_adv, cur_xent_adv = sess3.run(
                        [model3.num_correct,model3.xent],
                        feed_dict = dict_adv)
                    fmnist_loop_list[0,ibatch]=epsilonc
                    fmnist_loop_list[1,ibatch]=cur_corr_adv/eval_batch_size
                    fmnist_loop_list_corr[0,ibatch]=epsilonc
                    fmnist_loop_list_corr[1,ibatch]=cur_corr_nat/eval_batch_size
                    epsilonc+=config['epsilon']/num_batches
                    '''total_xent_nat += cur_xent_nat
                    total_xent_adv += cur_xent_adv
                    total_corr_nat += cur_corr_nat
                    total_corr_adv += cur_corr_adv
                    
            
                avg_xent_nat = total_xent_nat / num_eval_examples
                avg_xent_adv = total_xent_adv / num_eval_examples
                acc_nat = total_corr_nat / num_eval_examples
                acc_adv = total_corr_adv / num_eval_examples
                ''''''summary = tf.Summary(value=[
                tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
                tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
                tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
                tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
                tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
                tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
                summary_writer.add_summary(summary, global_step.eval(sess3))''''''
            #sys.stdout = sys.__stdout__
        print('natural: {:.2f}%'.format(100 * acc_nat))
        print('adversarial: {:.2f}%'.format(100 * acc_adv))
        print('avg nat loss: {:.4f}'.format(avg_xent_nat))
        print('avg adv loss: {:.4f}'.format(avg_xent_adv))'''
        np.save('loop_data/fmnist/fmnist_loop_list'+str(i+15)+".npy",fmnist_loop_list)
        np.save('loop_data/fmnist/fmnist_loop_list_corr'+str(i+15)+".npy",fmnist_loop_list_corr)
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


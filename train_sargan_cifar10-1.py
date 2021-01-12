import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
from sargan_dep.sargan_models import SARGAN
import os
from tqdm import tqdm
from random import shuffle
import skimage.measure as ski_me
import time
import numpy as np
import torch
from sargan_dep.cifar_helper import get_data, chunks
from sargan_dep.sar_utilities import add_gaussian_noise
from sargan_dep.alert_utilities import send_images_via_email

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

img_size = [28,28,1]
experiment_name = '/sargan_cifar10'
output_dir = 'outputs'
trained_model_path = 'trained_models/sargan_cifar10-1'
data_root='sar_data/cifar10'
output_path = output_dir+experiment_name
NUM_ITERATION = 85
BATCH_SIZE = 64
GPU_ID = 1
MAX_EPOCH = 55
LEARNING_RATE = 0.001
NOISE_STD_RANGE = [0.0, 0.4]
SAVE_EVERY_EPOCH = 5
plt.switch_backend('agg')

####
#GETTING IMAGES
####


def get_data(train_batch_size, val_batch_size):
    
    mnist = datasets.CIFAR10(root=data_root, train=True, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True)#.data.float()
    data_transform = Compose([Resize((img_size[0], img_size[1])),ToTensor()])
    
    train_loader = DataLoader(datasets.CIFAR10(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(datasets.CIFAR10(root=data_root, train=False, transform=data_transform, target_transform=None, download=True),
                            batch_size=val_batch_size, shuffle=False)
    
    
    return train_loader, val_loader




def main(args):
    image_number=0;
    model = SARGAN(img_size, BATCH_SIZE, img_channel=img_size[2])
    with tf.variable_scope("d_opt",reuse=tf.AUTO_REUSE):
        d_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.d_loss, var_list=model.d_vars)
    with tf.variable_scope("g_opt",reuse=tf.AUTO_REUSE):
        g_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.g_loss, var_list=model.g_vars)
    saver = tf.train.Saver(max_to_keep=20)
    
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(GPU_ID))
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    progress_bar = tqdm(range(MAX_EPOCH), unit="epoch")
    #list of loss values each item is the loss value of one ieteration
    train_d_loss_values = []
    train_g_loss_values = []
    
    
    #test_imgs, test_classes = get_data(test_filename)
    #imgs, classes = get_data(train_filename)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        #copies = imgs.astype('float32')
        #test_copies = test_imgs.astype('float32')
        for epoch in progress_bar:
            train_loader, val_loader = get_data(BATCH_SIZE, BATCH_SIZE)
            counter = 0
            epoch_start_time = time.time()
            #shuffle(copies)
            #divide the images into equal sized batches
            #image_batches = np.array(list(chunks(copies, BATCH_SIZE)))
            trainiter = iter(train_loader)
            for i in range (NUM_ITERATION):
                #getting a batch from the training data
                #one_batch_of_imgs = image_batches[i]
                features2, labels = next(trainiter)
                features2 = features2.data.numpy().transpose(0,2,3,1)*255
                
                
                features=np.zeros([len(features2),img_size[0],img_size[1],1])
                for i in range(len(features2)):
                    nextimage=Image.fromarray((features2[i]).astype(np.uint8))
                    nextimage=nextimage.convert('L')
                    features[i,:,:,0]=np.array(nextimage,dtype='float32')/255
                
                #copy the batch
                features_copy = features.copy()
                #corrupt the images
                corrupted_batch = np.array([add_gaussian_noise(image, sd=np.random.uniform(NOISE_STD_RANGE[1], NOISE_STD_RANGE[1])) for image in features_copy])
                for i in range(len(corrupted_batch)):
                    corrupted_batch[i]=gaussian_filter(corrupted_batch[i], sigma=1)
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:features, model.cond:corrupted_batch})
                _, M = sess.run([g_opt, model.g_loss], feed_dict={model.image:features, model.cond:corrupted_batch})
                train_d_loss_values.append(m)
                train_g_loss_values.append(M)
                #print some notifications
                counter += 1
                if counter % 25 == 0:
                    print("\rEpoch [%d], Iteration [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, counter, time.time() - epoch_start_time, m, M))
                
            # save the trained network
            if epoch % SAVE_EVERY_EPOCH == 0:
                save_path = saver.save(sess, (trained_model_path+"/sargan_mnist"))
                print("\n\nModel saved in file: %s\n\n" % save_path)
            ''' +\
                                   "%s_model_%s.ckpt" % ( experiment_name, epoch+1))'''
            
            
            ##### TESTING FOR CURRUNT EPOCH
            testiter = iter(val_loader)
            NUM_TEST_PER_EPOCH = 1
           
            #test_batches = np.array(list(chunks(test_copies, BATCH_SIZE)))
            #test_images = test_batches[0]
            sum_psnr = 0
            list_images = []
            for j in range(NUM_TEST_PER_EPOCH):
                features2, labels = next(trainiter)
                features2 = features2.data.numpy().transpose(0,2,3,1)*255
                
                
                features=np.zeros([len(features2),img_size[0],img_size[1],1])
                for i in range(len(features2)):
                    nextimage=Image.fromarray((features2[i]).astype(np.uint8))
                    nextimage=nextimage.convert('L')
                    features[i,:,:,0]=np.array(nextimage,dtype='float32')/255
                batch_copy = features.copy()
                #corrupt the images
                corrupted_batch = np.array([add_gaussian_noise(image, sd=np.random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1])) for image in batch_copy])
                for i in range(len(corrupted_batch)):
                    corrupted_batch[i]=gaussian_filter(corrupted_batch[i], sigma=1)
                

                gen_imgs = sess.run(model.gen_img, feed_dict={model.image:features, model.cond:corrupted_batch})
                print(features.shape, gen_imgs.shape)
                #if j %17 == 0: # only save 3 images 0, 17, 34
                list_images.append((features[0], corrupted_batch[0], gen_imgs[0]))
                list_images.append((features[17], corrupted_batch[17], gen_imgs[17]))                
                list_images.append((features[34], corrupted_batch[34], gen_imgs[34]))
                for i in range(len(gen_imgs)):
                    current_img = features[i]
                    recovered_img = gen_imgs[i]
                    sum_psnr += ski_me.compare_psnr(current_img, recovered_img, 1)
                #psnr_value = ski_mem.compare_psnr(test_img, gen_img, 1)
                #sum_psnr += psnr_value
            average_psnr = sum_psnr / 50
            
            epoch_running_time = time.time() - epoch_start_time
            ############### SEND EMAIL ##############
            rows = 1
            cols = 3
            display_mean = np.array([0.485, 0.456, 0.406])
            display_std = np.array([0.229, 0.224, 0.225])
            if epoch % SAVE_EVERY_EPOCH == 0: 
                #image = std * image + mean
                imgs_1 = list_images[0]
                imgs_2 = list_images[1]
                imgs_3 = list_images[2]
                imgs_1 = display_std * imgs_1 + display_mean
                imgs_2 = display_std * imgs_2 + display_mean
                imgs_3 = display_std * imgs_3 + display_mean
                fig = plt.figure(figsize=(14, 4))
                ax = fig.add_subplot(rows, cols, 1)
                ax.imshow(imgs_1[0])
                ax.set_title("Original", color='grey')
                ax = fig.add_subplot(rows, cols, 2)
                ax.imshow(imgs_1[1])
                ax.set_title("Corrupted", color='grey')
                ax = fig.add_subplot(rows, cols, 3)
                ax.imshow(imgs_1[2])
                ax.set_title("Recovered", color='grey')
                plt.tight_layout()
                #sample_test_file_1 = os.path.join(output_path, '%s_epoch_%s_batchsize_%s_1.jpg' % (experiment_name, epoch, BATCH_SIZE))
                sample_test_file_1 = os.path.join(output_path, 'image_%d_1.jpg' % image_number)
                plt.savefig(sample_test_file_1, dpi=300)
                
                fig = plt.figure(figsize=(14, 4))
                ax = fig.add_subplot(rows, cols, 1)
                ax.imshow(imgs_2[0])
                ax.set_title("Original", color='grey')
                ax = fig.add_subplot(rows, cols, 2)
                ax.imshow(imgs_2[1])
                ax.set_title("Corrupted", color='grey')
                ax = fig.add_subplot(rows, cols, 3)
                ax.imshow(imgs_2[2])
                ax.set_title("Recovered", color='grey')
                plt.tight_layout()
                #sample_test_file_2 = os.path.join(output_path, '%s_epoch_%s_batchsize_%s_2.jpg' % (experiment_name, epoch, BATCH_SIZE))
                sample_test_file_2 = os.path.join(output_path, 'image_%d_2.jpg' % image_number)
                plt.savefig(sample_test_file_2, dpi=300)
     
                fig = plt.figure(figsize=(14, 4))
                ax = fig.add_subplot(rows, cols, 1)
                ax.imshow(imgs_3[0])
                ax.set_title("Original", color='grey')
                ax = fig.add_subplot(rows, cols, 2)
                ax.imshow(imgs_3[1])
                ax.set_title("Corrupted", color='grey')
                ax = fig.add_subplot(rows, cols, 3)
                ax.imshow(imgs_3[2])
                ax.set_title("Recovered", color='grey')
                plt.tight_layout()
                #sample_test_file_3 = os.path.join(output_path, '%s_epoch_%s_batchsize_%s_3.jpg' % (experiment_name, epoch, BATCH_SIZE))
                sample_test_file_3 = os.path.join(output_path, 'image_%d_3.jpg' % image_number)
                image_number+=1
                plt.savefig(sample_test_file_3, dpi=300)
            plt.close("all")            
        
if __name__ == '__main__':
    main([])

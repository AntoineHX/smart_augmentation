import matplotlib.pyplot as plt
from far_ho.examples.datasets import Datasets, Dataset

import os
import numpy as np
import tensorflow as tf

import augmentation_transforms as augmentation_transforms ##### ATTENTION FICHIER EN DOUBLE => A REGLER MIEUX ####

def viz_data(dataset, fig_name='data_sample',aug_policy=None):

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        img = dataset.data[i][:,:,0]
        if aug_policy :
            img = augment_img(img,aug_policy)
        #print('im shape',img.shape)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel(np.nonzero(dataset.target[i])[0].item())

    plt.savefig(fig_name)

def augment_img(data, policy):

    #print('Im shape',data.shape)
    data = np.stack((data,)*3, axis=-1) #BOF BOF juste pour forcer 3 channels
    #print('Im shape',data.shape)
    final_img = augmentation_transforms.apply_policy(policy, data)
    #final_img = augmentation_transforms.random_flip(augmentation_transforms.zero_pad_and_crop(final_img, 4))
    # Apply cutout
    #final_img = augmentation_transforms.cutout_numpy(final_img)
    
    im_rgb = np.array(final_img, np.float32)
    im_gray = np.dot(im_rgb[...,:3], [0.2989, 0.5870, 0.1140]) #Just pour retourner a 1 channel

    return im_gray


### https://www.kaggle.com/raoulma/mnist-image-class-tensorflow-cnn-99-51-test-acc#5.-Build-the-neural-network-with-tensorflow-
## build the neural network class
# weight initialization
def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

# bias initialization
def bias_variable(shape, name = None):
    initial = tf.constant(0.1, shape=shape) #  positive bias
    return tf.Variable(initial, name = name)

# 2D convolution
def conv2d(x, W, name = None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

# max pooling
def max_pool_2x2(x, name = None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name = name)

def cnn(x_data_tf,y_data_tf, name='model'):
     # tunable hyperparameters for nn architecture
     s_f_conv1 = 3; # filter size of first convolution layer (default = 3)
     n_f_conv1 = 36; # number of features of first convolution layer (default = 36)
     s_f_conv2 = 3; # filter size of second convolution layer (default = 3)
     n_f_conv2 = 36; # number of features of second convolution layer (default = 36)
     s_f_conv3 = 3; # filter size of third convolution layer (default = 3)
     n_f_conv3 = 36; # number of features of third convolution layer (default = 36)
     n_n_fc1 = 576; # number of neurons of first fully connected layer (default = 576)

     # 1.layer: convolution + max pooling
     W_conv1_tf = weight_variable([s_f_conv1, s_f_conv1, 1, n_f_conv1], name = 'W_conv1_tf') # (5,5,1,32)
     b_conv1_tf = bias_variable([n_f_conv1], name = 'b_conv1_tf') # (32)
     h_conv1_tf = tf.nn.relu(conv2d(x_data_tf, 
                                                 W_conv1_tf) + b_conv1_tf, 
                                     name = 'h_conv1_tf') # (.,28,28,32)
     h_pool1_tf = max_pool_2x2(h_conv1_tf, 
                                            name = 'h_pool1_tf') # (.,14,14,32)

     # 2.layer: convolution + max pooling
     W_conv2_tf = weight_variable([s_f_conv2, s_f_conv2, 
                                                n_f_conv1, n_f_conv2], 
                                               name = 'W_conv2_tf')
     b_conv2_tf = bias_variable([n_f_conv2], name = 'b_conv2_tf')
     h_conv2_tf = tf.nn.relu(conv2d(h_pool1_tf, 
                                                 W_conv2_tf) + b_conv2_tf, 
                                     name ='h_conv2_tf') #(.,14,14,32)
     h_pool2_tf = max_pool_2x2(h_conv2_tf, name = 'h_pool2_tf') #(.,7,7,32)

     # 3.layer: convolution + max pooling
     W_conv3_tf = weight_variable([s_f_conv3, s_f_conv3, 
                                                n_f_conv2, n_f_conv3], 
                                               name = 'W_conv3_tf')
     b_conv3_tf = bias_variable([n_f_conv3], name = 'b_conv3_tf')
     h_conv3_tf = tf.nn.relu(conv2d(h_pool2_tf, 
                                                 W_conv3_tf) + b_conv3_tf, 
                                     name = 'h_conv3_tf') #(.,7,7,32)
     h_pool3_tf = max_pool_2x2(h_conv3_tf, 
                                            name = 'h_pool3_tf') # (.,4,4,32)

     # 4.layer: fully connected
     W_fc1_tf = weight_variable([4*4*n_f_conv3,n_n_fc1], 
                                             name = 'W_fc1_tf') # (4*4*32, 1024)
     b_fc1_tf = bias_variable([n_n_fc1], name = 'b_fc1_tf') # (1024)
     h_pool3_flat_tf = tf.reshape(h_pool3_tf, [-1,4*4*n_f_conv3], 
                                          name = 'h_pool3_flat_tf') # (.,1024)
     h_fc1_tf = tf.nn.relu(tf.matmul(h_pool3_flat_tf, 
                                             W_fc1_tf) + b_fc1_tf, 
                                   name = 'h_fc1_tf') # (.,1024)
      
     # add dropout
     #keep_prob_tf = tf.placeholder(dtype=tf.float32, name = 'keep_prob_tf')
     #h_fc1_drop_tf = tf.nn.dropout(h_fc1_tf, keep_prob_tf, name = 'h_fc1_drop_tf')

     # 5.layer: fully connected
     W_fc2_tf = weight_variable([n_n_fc1, 10], name = 'W_fc2_tf')
     b_fc2_tf = bias_variable([10], name = 'b_fc2_tf')
     z_pred_tf = tf.add(tf.matmul(h_fc1_tf, W_fc2_tf), 
                                b_fc2_tf, name = 'z_pred_tf')# => (.,10)
     # predicted probabilities in one-hot encoding
     y_pred_proba_tf = tf.nn.softmax(z_pred_tf, name='y_pred_proba_tf') 
        
     # tensor of correct predictions
     y_pred_correct_tf = tf.equal(tf.argmax(y_pred_proba_tf, 1),
                                          tf.argmax(y_data_tf, 1),
                                          name = 'y_pred_correct_tf')  
     return y_pred_proba_tf
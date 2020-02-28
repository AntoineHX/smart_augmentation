import numpy as np
import tensorflow as tf

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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name = name)

# max pooling
def max_pool_2x2(x, name = None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name = name)

def LeNet(images, num_classes):
     # tunable hyperparameters for nn architecture
     s_f_conv1 = 5; # filter size of first convolution layer (default = 3)
     n_f_conv1 = 20; # number of features of first convolution layer (default = 36)
     s_f_conv2 = 5; # filter size of second convolution layer (default = 3)
     n_f_conv2 = 50; # number of features of second convolution layer (default = 36)
     n_n_fc1 = 500; # number of neurons of first fully connected layer (default = 576)
     n_n_fc2 = 500; # number of neurons of first fully connected layer (default = 576)

     #print(images.shape)
     # 1.layer: convolution + max pooling
     W_conv1_tf = weight_variable([s_f_conv1, s_f_conv1, int(images.shape[3]), n_f_conv1], name = 'W_conv1_tf') # (5,5,1,32)
     b_conv1_tf = bias_variable([n_f_conv1], name = 'b_conv1_tf') # (32)
     h_conv1_tf = tf.nn.relu(conv2d(images, W_conv1_tf) + b_conv1_tf, name = 'h_conv1_tf') # (.,28,28,32)
     h_pool1_tf = max_pool_2x2(h_conv1_tf, name = 'h_pool1_tf') # (.,14,14,32)
     #print(h_conv1_tf.shape)
     #print(h_pool1_tf.shape)
     # 2.layer: convolution + max pooling
     W_conv2_tf = weight_variable([s_f_conv2, s_f_conv2, n_f_conv1, n_f_conv2], name = 'W_conv2_tf')
     b_conv2_tf = bias_variable([n_f_conv2], name = 'b_conv2_tf')
     h_conv2_tf = tf.nn.relu(conv2d(h_pool1_tf, W_conv2_tf) + b_conv2_tf, name ='h_conv2_tf') #(.,14,14,32)
     h_pool2_tf = max_pool_2x2(h_conv2_tf, name = 'h_pool2_tf') #(.,7,7,32)

     #print(h_pool2_tf.shape)

     # 4.layer: fully connected
     W_fc1_tf = weight_variable([5*5*n_f_conv2,n_n_fc1], name = 'W_fc1_tf') # (4*4*32, 1024)
     b_fc1_tf = bias_variable([n_n_fc1], name = 'b_fc1_tf') # (1024)
     h_pool2_flat_tf = tf.reshape(h_pool2_tf, [int(h_pool2_tf.shape[0]), -1], name = 'h_pool3_flat_tf') # (.,1024)
     h_fc1_tf = tf.nn.relu(tf.matmul(h_pool2_flat_tf, W_fc1_tf) + b_fc1_tf, 
                                   name = 'h_fc1_tf') # (.,1024)
      
     # add dropout
     #keep_prob_tf = tf.placeholder(dtype=tf.float32, name = 'keep_prob_tf')
     #h_fc1_drop_tf = tf.nn.dropout(h_fc1_tf, keep_prob_tf, name = 'h_fc1_drop_tf')
     print(h_fc1_tf.shape)

     # 5.layer: fully connected
     W_fc2_tf = weight_variable([n_n_fc1, num_classes], name = 'W_fc2_tf')
     b_fc2_tf = bias_variable([num_classes], name = 'b_fc2_tf')
     z_pred_tf = tf.add(tf.matmul(h_fc1_tf, W_fc2_tf), b_fc2_tf, name = 'z_pred_tf')# => (.,10)
     # predicted probabilities in one-hot encoding
     #y_pred_proba_tf = tf.nn.softmax(z_pred_tf, name='y_pred_proba_tf') 
        
     # tensor of correct predictions
     #y_pred_correct_tf = tf.equal(tf.argmax(y_pred_proba_tf, 1),
     #                                     tf.argmax(y_data_tf, 1),
     #                                     name = 'y_pred_correct_tf')  
     logits = z_pred_tf
     return logits #y_pred_proba_tf

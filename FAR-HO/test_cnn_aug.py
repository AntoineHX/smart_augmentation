import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import far_ho as far
import far_ho.examples as far_ex

tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib.pyplot as plt
import blue_utils as butil

#Reset
try:
    sess.close()
except: pass
rnd = np.random.RandomState(1)
tf.reset_default_graph()
sess = tf.InteractiveSession()

def get_data(data_split):
    # load a small portion of mnist data
    datasets = far_ex.mnist(data_root_folder=os.path.join(os.getcwd(), 'MNIST_DATA'), partitions=data_split, reshape=False)
    print("Data shape : ", datasets.train.dim_data, "/ Label shape : ", datasets.train.dim_target)
    [print("Nb samples : ", d.num_examples) for d in datasets]
    return datasets.train, datasets.validation, datasets.test

#Model
# FC : reshape = True
def g_logits(x,y, name='model'):
    with tf.variable_scope(name):
        h1 = layers.fully_connected(x, 300)
        logits = layers.fully_connected(h1, int(y.shape[1]))
    return logits

#### Hyper-parametres ####
n_hyper_iterations = 10
T = 10 # Number of inner iterations
rev_it =10
hp_lr = 0.02
##########################

#MNIST
#x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
#y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
#logits = g_logits(x, y)

#CNN : reshape = False
x = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None,10], name='y')

logits = butil.cnn(x,y)

train_set, validation_set, test_set = get_data(data_split=(.1, .1,))

probX = far.get_hyperparameter('probX', initializer=0.1, constraint=lambda t: tf.maximum(tf.minimum(t, 0.1), 0.9))
probY = far.get_hyperparameter('probY', initializer=0.1, constraint=lambda t: tf.maximum(tf.minimum(t, 0.1), 0.9))

#lr = far.get_hyperparameter('lr', initializer=1e-4, constraint=lambda t: tf.maximum(tf.minimum(t, 1e-4), 1e-4))
#mu = far.get_hyperparameter('mu', initializer=0.9, constraint=lambda t: tf.maximum(tf.minimum(t, 0.9), 0.9))

#probX, probY = 0.5, 0.5
#policy = [('TranslateX', probX, 8), ('TranslateY', probY, 8)]
policy = [('TranslateX', probX, 8), ('FlipUD', probY, 8)]
print('Hyp :',far.utils.hyperparameters(scope=None))

#butil.viz_data(train_set, aug_policy= policy)
#print('Data sampled !')

#Ajout artificiel des transfo a la loss juste pour qu il soit compter dans la dynamique du graph
probX_loss = tf.sigmoid(probX)
probY_loss = tf.sigmoid(probY)

ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
L = tf.reduce_mean(probX_loss*probY_loss*ce)
E = tf.reduce_mean(ce)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32))

inner_optimizer = far.AdamOptimizer()
outer_optimizer = tf.train.AdamOptimizer(hp_lr)
hyper_method = far.ReverseHG().truncated(reverse_iterations=rev_it)
hyper_step = far.HyperOptimizer(hyper_method).minimize(E, outer_optimizer, L, inner_optimizer)

train_set_supplier = train_set.create_supplier(x, y, batch_size=256, aug_policy=policy)  # stochastic GD
validation_set_supplier = validation_set.create_supplier(x, y)

#print(train_set.dim_data,validation_set.dim_data)

his_params = []

tf.global_variables_initializer().run()

butil.viz_data(train_set, fig_name= 'Start_sample',aug_policy= policy)
print('Data sampled !')

for hyt in range(n_hyper_iterations):
    hyper_step(T,
               inner_objective_feed_dicts=train_set_supplier,
               outer_objective_feed_dicts=validation_set_supplier,
               _skip_hyper_ts=True)
    res = sess.run(far.hyperparameters()) + [L.eval(train_set_supplier()), 
                                             E.eval(validation_set_supplier()),
                                             accuracy.eval(train_set_supplier()),
                                             accuracy.eval(validation_set_supplier())]
    his_params.append(res)

    butil.viz_data(train_set, fig_name= 'Train_sample_{}'.format(hyt),aug_policy= policy)
    print('Data sampled !')

    print('Hyper-it :',hyt,'/',n_hyper_iterations)
    print('inner:', L.eval(train_set_supplier()))
    print('outer:', E.eval(validation_set_supplier()))
    print('training accuracy:', res[4])
    print('validation accuracy:', res[5])
    print('Transformation : ProbX -',res[0],'/ProbY -',res[1])
    #print('learning rate', lr.eval(), 'momentum', mu.eval(), 'l2 coefficient', rho.eval())
    print('-'*50)

test_set_supplier = test_set.create_supplier(x, y)
print('Test accuracy:',accuracy.eval(test_set_supplier()))

fig, ax = plt.subplots(ncols=4, figsize=(15, 3))
ax[0].set_title('ProbX')
ax[0].plot([e[0] for e in his_params])
    
ax[1].set_title('ProbY')
ax[1].plot([e[1] for e in his_params]) 
    
ax[2].set_title('Tr. and val. errors')
ax[2].plot([e[2] for e in his_params])
ax[2].plot([e[3] for e in his_params])  

ax[3].set_title('Tr. and val. acc')
ax[3].plot([e[4] for e in his_params])
ax[3].plot([e[5] for e in his_params])

plt.savefig('res_cnn_aug_H{}_I{}'.format(n_hyper_iterations,T))

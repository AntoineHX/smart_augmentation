#https://github.com/lucfra/FAR-HO/blob/master/far_ho/examples/autoMLDemos/Far-HO%20Demo%2C%20AutoML%202018%2C%20ICML%20workshop.ipynb
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
#import blue_utils as butil

#Reset
try:
    sess.close()
except: pass
rnd = np.random.RandomState(1)
tf.reset_default_graph()
sess = tf.InteractiveSession()

def get_data(data_split):
    # load a small portion of mnist data
    datasets = far_ex.mnist(data_root_folder=os.path.join(os.getcwd(), 'MNIST_DATA'), partitions=data_split, reshape=True)
    print("Data shape : ", datasets.train.dim_data, " / Label shape : ", datasets.train.dim_target)
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
n_hyper_iterations = 90
T = 20  # Number of inner iterations
rev_it =10
hp_lr = 0.1
epochs =10
batch_size = 256
##########################

#MNIST
x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
logits = g_logits(x, y)

#CNN : reshape = False
#x = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1], name='x')
#y = tf.placeholder(dtype=tf.float32, shape=[None,10], name='y')

#logits = butil.cnn(x,y)

train_set, validation_set, test_set = get_data(data_split=(.6, .3,))

#butil.viz_data(train_set)

# lambdas = far.get_hyperparameter('lambdas', tf.zeros(train_set.num_examples))
lr = far.get_hyperparameter('lr', initializer=1e-2, constraint=lambda t: tf.maximum(tf.minimum(t, 0.1), 1.e-7))
mu = far.get_hyperparameter('mu', initializer=0.95, constraint=lambda t: tf.maximum(tf.minimum(t, .99), 1.e-5))
#rho = far.get_hyperparameter('rho', initializer=0.00001, constraint=lambda t: tf.maximum(tf.minimum(t, 0.01), 0.))


ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
L = tf.reduce_mean(ce) #+ rho*tf.add_n([tf.reduce_sum(w**2) for w in tf.trainable_variables()]) #Retirer la seconde partie de la loss quand HP inutiles
E = tf.reduce_mean(ce)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32))

inner_optimizer = far.MomentumOptimizer(lr, mu)
#inner_optimizer = far.GradientDescentOptimizer(lr)
outer_optimizer = tf.train.AdamOptimizer(hp_lr)
hyper_method = far.ReverseHG().truncated(reverse_iterations=rev_it)
hyper_step = far.HyperOptimizer(hyper_method).minimize(E, outer_optimizer, L, inner_optimizer)#, global_step=tf.train.get_or_create_step())

train_set_supplier = train_set.create_supplier(x, y, batch_size=batch_size)#, epochs=1)  # stochastic GD
validation_set_supplier = validation_set.create_supplier(x, y)


print('Hyper iterations par epochs',int(train_set.num_examples/batch_size*epochs/T))

his_params = []

tf.global_variables_initializer().run()

for hyt in range(n_hyper_iterations):
    hyper_step(T,
               inner_objective_feed_dicts=train_set_supplier,
               outer_objective_feed_dicts=validation_set_supplier,
               _skip_hyper_ts=False)
    res = sess.run(far.hyperparameters()) + [0, L.eval(train_set_supplier()), 
                                             E.eval(validation_set_supplier()),
                                             accuracy.eval(train_set_supplier()),
                                             accuracy.eval(validation_set_supplier())]

    his_params.append(res)

    print('Hyper-it :',hyt,'/',n_hyper_iterations)
    print('inner:', res[3])
    print('outer:', res[4])
    print('training accuracy:', res[5])
    print('validation accuracy:', res[6])
    #print('learning rate', lr.eval(), 'momentum', mu.eval(), 'l2 coefficient', rho.eval())
    print('-'*50)

test_set_supplier = test_set.create_supplier(x, y)
print('Test accuracy:',accuracy.eval(test_set_supplier()))

fig, ax = plt.subplots(ncols=4, figsize=(15, 3))
ax[0].set_title('Learning rate')
ax[0].plot([e[0] for e in his_params])
    
ax[1].set_title('Momentum factor')
ax[1].plot([e[1] for e in his_params]) 
    
#ax[2].set_title('L2 regulariz.')
#ax[2].plot([e[2] for e in his_params])
ax[2].set_title('Tr. and val. acc')
ax[2].plot([e[5] for e in his_params])
ax[2].plot([e[6] for e in his_params])
    
ax[3].set_title('Tr. and val. errors')
ax[3].plot([e[3] for e in his_params])
ax[3].plot([e[4] for e in his_params])  

plt.savefig('resultats/res_fc_H{}_I{}'.format(n_hyper_iterations,T))
#plt.savefig('resultats/res_fc_H{}_I{}_noHyp'.format(n_hyper_iterations,T))

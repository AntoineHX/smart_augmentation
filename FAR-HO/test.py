import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import far_ho as far
import far_ho.examples as far_ex
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()


def get_data():
    # load a small portion of mnist data
    datasets = far_ex.mnist(data_root_folder=os.path.join(os.getcwd(), 'MNIST_DATA'), partitions=(.1, .1,))
    return datasets.train, datasets.validation


def g_logits(x,y):
    with tf.variable_scope('model'):
        h1 = layers.fully_connected(x, 300)
        logits = layers.fully_connected(h1, int(y.shape[1]))
    return logits


x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
logits = g_logits(x,y)
train_set, validation_set = get_data()

lambdas = far.get_hyperparameter('lambdas', tf.zeros(train_set.num_examples))
lr = far.get_hyperparameter('lr', initializer=0.01)

ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
L = tf.reduce_mean(tf.sigmoid(lambdas)*ce)
E = tf.reduce_mean(ce)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32))

inner_optimizer = far.GradientDescentOptimizer(lr)
outer_optimizer = tf.train.AdamOptimizer()
rev_it =10
hyper_method = far.ReverseHG().truncated(reverse_iterations=rev_it)
hyper_step = far.HyperOptimizer(hyper_method).minimize(E, outer_optimizer, L, inner_optimizer)

T = 20  # Number of inner iterations
train_set_supplier = train_set.create_supplier(x, y)
validation_set_supplier = validation_set.create_supplier(x, y)
tf.global_variables_initializer().run()

print('inner:', L.eval(train_set_supplier()))
print('outer:', E.eval(validation_set_supplier()))
# print('-'*50)
n_hyper_iterations = 200
inner_losses = []
outer_losses = []
train_accs = []
val_accs = []

for _ in range(n_hyper_iterations):
    hyper_step(T,
               inner_objective_feed_dicts=train_set_supplier,
               outer_objective_feed_dicts=validation_set_supplier)

    inner_obj = L.eval(train_set_supplier())
    outer_obj = E.eval(validation_set_supplier())
    inner_losses.append(inner_obj)
    outer_losses.append(outer_obj)
    print('inner:', inner_obj)
    print('outer:', outer_obj)

    train_acc = accuracy.eval(train_set_supplier())
    val_acc = accuracy.eval(validation_set_supplier())
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print('training accuracy', train_acc)
    print('validation accuracy', val_acc)

    print('learning rate', lr.eval())
    print('norm of examples weight', tf.norm(lambdas).eval())
    print('-'*50)
    
plt.subplot(211)
plt.plot(inner_losses, label='training loss')
plt.plot(outer_losses, label='validation loss')
plt.legend(loc=0, frameon=True)
#plt.xlim(0, 19)
plt.subplot(212)
plt.plot(train_accs, label='training accuracy')
plt.plot(val_accs, label='validation accuracy')
plt.legend(loc=0, frameon=True)

plt.savefig('H%d - I%d - R%d'%(n_hyper_iterations,T,rev_it))

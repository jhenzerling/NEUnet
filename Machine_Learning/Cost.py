#This module contains the COST functions
#Cost functions allow improvement via machine learning

import tensorflow as tf


#MNIST Cost Setup and Getters
def MNIST_Cost_Setup(y,network,learningrate):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=network))
    train_step = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)   
    correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return [cross_entropy,train_step,accuracy]

#CIFAR Cost Setup and Getters
def CIFAR10_Cost_Setup(y,network,learningrate):

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=network))
    step = tf.Variable(0, trainable = False, name = 'step')
    rate = tf.train.exponential_decay(learningrate, step, 1, .95)
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy, global_step = step)
    correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return [cross_entropy,train_step,accuracy]

def PSingle_Cost_Setup(y,network,learningrate):
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=network))
	train_step = tf.train.RMSPropOptimizer(learningrate).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(y, 1))
  	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	

	return [cross_entropy, train_step, accuracy]

#Use these as a way to edit the cost function and training steps/accuracies

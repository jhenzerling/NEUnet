#MNIST Data Module

import tensorflow as tf
import os

#MNIST Libraries for Numbers
if(os.path.exists("./Data/MNIST/")):
    print('MNIST Data Present.')
else:
    print('MNIST Data Not Present. Downloading.')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Data/MNIST', one_hot=True)

#Image Dimensions and Colours
hsize = 28
vsize = 28
colours = 1
#Training Data Size, Batch Size, and Test Data Size
#Batchsize designates number of points in each separate chunk
#Stepnumber designates number batches to go over
stepnumber = 100
batchsize = 512
testsize = 10000
#Class Number
cnumb = 10

#Assign Batches
with tf.variable_scope('MNIST_Data'):
    t1,t2 = mnist.train.next_batch(batchsize)
    trainb = [tf.reshape(t1,[batchsize,hsize,vsize,colours]),tf.reshape(t2,[batchsize,cnumb])]
    testb = [tf.reshape(mnist.test.images,[testsize,hsize,vsize,1]), tf.reshape(mnist.test.labels,[testsize,cnumb])]

#Getters
def get_dataparams():
    return [hsize,vsize,colours,cnumb]

def get_trainparams():
    return [stepnumber,batchsize,testsize]

def get_batch():
    return [trainb,testb]
    

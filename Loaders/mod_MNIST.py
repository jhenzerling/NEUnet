#MNIST Data Module
import tensorflow as tf
import os
import numpy as np
#Configuration Files
from Configs import config_data as cd
from Configs import config_main as cm

#Path
path 		= cd.MNIST['Path']
#Check for data
if(os.listdir(path) != []):
    print('MNIST Data Present.')
else:
    print('MNIST Data Not Present.')
    quit()
    
#Pull data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Data/MNIST', one_hot=True)

#Image Dimensions and Colours and Classes and Axes
hsize 		= cd.MNIST['H']
vsize 		= cd.MNIST['V']
colours 	= cd.MNIST['Colours']
cnumb 		= cd.MNIST['Classes']
axes 		= cd.MNIST['Axes']
#Size of the dataset
trainsize 	= cd.MNIST['Train']
testsize 	= cd.MNIST['Test']
#Batchsize designates number of points in each separate chunk
#Stepnumber designates number batches to go over
stepnumber 	= cm.LEARN['Step']
batchsize 	= cm.LEARN['Batch']


#Assign Batches
with tf.variable_scope('MNIST_Data'):
    t1,t2 	= mnist.train.next_batch(batchsize)
    trainbatch  = [tf.reshape(t1,[batchsize,hsize,vsize,colours]),tf.reshape(t2,[batchsize,cnumb])]
    testbatch   = [tf.reshape(mnist.test.images,[testsize,hsize,vsize,1]),
			 	tf.reshape(mnist.test.labels,[testsize,cnumb])]

##################################################################
#Bin the spread of elements in classes
def spread():
	trlab = mnist.train.labels
	telab = mnist.test.labels
	trsum = np.zeros(cnumb)
	tesum = np.zeros(cnumb)	
	for x in range(len(trlab)):
		trsum += trlab[x]
		if x < len(telab):
			tesum += telab[x]
	return [trsum,tesum]

		



    

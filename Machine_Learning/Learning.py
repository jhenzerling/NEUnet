#This module groups up the learning techniques so it's easier

#Numpy for number-handling
import numpy as np

#File Handling
from Logging import mod_log as log
import sys
import time
import datetime
time.clock()

#Machine Learning Handling
from Machine_Learning import Cost as cost
from Machine_Learning import Network as network

#Tensorflow Libraries
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#LARCV
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

#Main Configuration
from Configs import config_main as config

#Set up MNIST/CIFAR Feeds

def MNIST_Feeds(a,b,c,d):
	Qtrainbatch0,Qtrainbatch1 = sess.run([a,b])
        Qtestbatch0,Qtestbatch1 = sess.run([a,b])
	trainfeed = {c: Qtrainbatch0, d: Qtrainbatch1, keep_prob: 0.8}
        testfeed = {d: Qtestbatch0, d: Qtestbatch1, keep_prob: 1.0}

	return [trainfeed,testfeed]

def CIFAR10_Feeds(a,b,c,d):
	Qtrainbatch0,Qtrainbatch1 = sess.run([a,b])
        Qtestbatch0,Qtestbatch1 = sess.run([a,b])
	trainfeed = {c: Qtrainbatch0, d: Qtrainbatch1, keep_prob: 0.8}
        testfeed = {d: Qtestbatch0, d: Qtestbatch1, keep_prob: 1.0}

	return [trainfeed,testfeed]

#Set up PSingle Feed
def PSingle_Feeds(a,c,d,b):
	if b == 'train':
		trdata = a.fetch_data('train_image').data()
        	trlabel = a.fetch_data('train_label').data()
	elif b == 'test':
		trdata = a.fetch_data('test_image').data()
        	trlabel = a.fetch_data('test_label').data()
	else:
		print('Bad type')
        trainfeed = {c: trdata, d: trlabel}
	
	return trainfeed









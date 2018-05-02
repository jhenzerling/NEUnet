#Module for TensorBoard Functionality for simplicity

import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from larcv import larcv
import ROOT

#Set up the histograms
def setHistograms(name):
	for var in tf.trainable_variables():
        	tf.summary.histogram(var.name,var,collections=[name])
	return

#Set up the trainer storings
def setSummaries(sess):

        merge = tf.summary.merge_all()
	tf.trainable_variables(tf.GraphKeys.GLOBAL_VARIABLES)
        writertrain=tf.summary.FileWriter('Graphs/Train', sess.graph)
        writertrain.add_graph(sess.graph)
        writertest=tf.summary.FileWriter('Graphs/Test', sess.graph)
        writertest.add_graph(sess.graph)

	return [writertrain,writertest,merge]

#Print the helper
def TBPrint():
	print('CALL THE FOLLOWING COMMAND FOR TENSORBOARD: ')
        print('tensorboard --logdir=run1:/user/jhenzerling/work/neunet/Graphs/ --port 8008')
	return

#Feed into Summary
def SummFeeder(merge,accu,sess,trainfeed,testfeed):
	summarytrain,acctrain = sess.run([merge,accu], feed_dict=trainfeed)
	summarytest,acctest = sess.run([merge,accu], feed_dict=testfeed)
	return [summarytrain,acctrain,summarytest,acctest]

#Add to Summary
def SummAdder(writertrain,writertest,summarytrain,summarytest,i):
	writertrain.add_summary(summarytrain,i)
	writertest.add_summary(summarytest,i)
	return
	
#Close out the threads for train and test
def WCloser(writertrain,writertest):
	writertrain.close()
	writertest.close()
	return

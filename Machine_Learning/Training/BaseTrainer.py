#This module groups up the learning techniques so it's easier
#Numpy for number-handling
import numpy as np

#File Handling
from Logging import mod_log as log
from Logging import mod_graph as graph
import sys,os,time

#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Machine Learning Handling
from Machine_Learning import Learning as learn
start = time.time()

#Tensorflow Libraries
import tensorflow as tf


#Training Step for NON PROC data
class BaseTrainer:
	#Instantiate Globals
	def __init__(self,trbatch,tebatch,x,y_,keep_prob,accuracy,costfunction,train_step):
		self.trainbatch = trbatch
		self.testbatch = tebatch
		self.x = x
		self.y_ = y_
		self.keep_prob = keep_prob
		self.accuracy = accuracy
		self.costfunction = costfunction
		self.train_step = train_step
	#Running the session with configuration as follows
	def starter(self):
		config = tf.ConfigProto()
		#config.gpu_options.per_process_gpu_memory_fraction = 0.7
		config.gpu_options.allow_growth = True
		return config
	#Set Threads
	def threader(self):	
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)
		return [coord,threads]
	#Store Graphs
	def grapher(self,sess):
		graph.TBPrint()
		tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		tf.summary.scalar("Cost Function", self.costfunction)
        	tf.summary.scalar("Accuracy", self.accuracy)
		tf.summary.image('train_input',self.trainbatch[0],10)
                [writertrain,writertest,merged] = graph.setSummaries(sess)
                saver = tf.train.Saver()
		return [writertrain,writertest,merged,saver]
	#TrainingStep for setting Feeds
	def nextstep(self,trigger,sess):
            	if trigger == 'CIFAR10':
                	[trainfeed,testfeed] = learn.CIFAR10_Feeds(self.trainbatch[0],self.trainbatch[1],self.x,self.y_,sess,self.keep_prob)
			return [trainfeed,testfeed]
            	elif trigger == 'MNIST':
                	[trainfeed,testfeed] = learn.MNIST_Feeds(self.trainbatch[0],self.trainbatch[1],self.x,self.y_,sess,self.keep_prob)
			return [trainfeed,testfeed]
            	else:
               		print('Bad Train and/or Test Feeds')
			quit()		
	#Sets Losses asd Accuracies
	def logger(self,trigger,merged,GT,sess,trainfeed,testfeed,writertrain,writertest,i,steps):

                #Graph Operations
                if GT == True:
                	[summarytrain,acctrain,summarytest,acctest] = graph.SummFeeder(merged,self.accuracy,sess,trainfeed,testfeed)
                    	graph.SummAdder(writertrain,writertest,summarytrain,summarytest,i)
                        

		train_accuracy 		= self.accuracy.eval(feed_dict=trainfeed)
                test_accuracy 		= self.accuracy.eval(feed_dict=testfeed)
                train_loss 		= self.costfunction.eval(feed_dict=trainfeed)
                test_loss 		= self.costfunction.eval(feed_dict=testfeed)
                log.outputprint(steps,train_accuracy,test_accuracy,
                                    train_loss,test_loss,start - time.time(),i)
	#Closes Threads
	def closer(self,coord,threads,writertrain,writertest,GT):
		coord.request_stop()
        	coord.join(threads)
        	if GT == True:
            		graph.WCloser(writertrain,writertest)
	#Now the function that handles training
	def training(self,trigger,steps,GT,logs):
		#Start the config
		config = self.starter()
		with tf.Session(config=config) as sess:
			[coord,threads] = self.threader()
			with tf.variable_scope('Session'):
				init = tf.global_variables_initializer()
				sess.run(init)
			#Log Graph
			if GT == True:
				[writertrain,writertest,merged,saver] = self.grapher(sess)
			for i in range(steps):
				[trainfeed,testfeed] = self.nextstep(trigger,sess)
				with tf.variable_scope('Feeding'):
					if i % (steps/logs) == 0:
						self.logger(trigger,merged,GT,sess,trainfeed,testfeed,writertrain,writertest,i,steps)
				with tf.variable_scope('Training'):
					self.train_step.run(feed_dict=trainfeed)
			self.closer(coord,threads,writertrain,writertest,GT)






























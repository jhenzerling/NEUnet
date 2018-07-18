#This module groups up the learning techniques so it's easier
#Numpy for number-handling
import numpy as np

#File Handling
from Logging import mod_log as log
from Logging import mod_graph as graph
import sys,os,time,datetime

#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Start Clock for timing
start = time.time()

#Data Handling
from Loaders import mod_dataobj as d
import importlib as i

#Machine Learning Handling
from Machine_Learning import Cost as cost
from Machine_Learning import Network as network
from Machine_Learning import Learning as learn

#Tensorflow Libraries
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#LARCV
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

#Main Configuration
from Configs import config_main as cm

tf.reset_default_graph()

#Object that handles the training steps whatever
class Trainer:
	#Instantiate by pulling the information
	def __init__(self,data,tensors):
		self.DT			= cm.MAIN['Data']
		self.GT			= cm.MAIN['Graph']
		self.LOG 		= cm.MAIN['Logging']
		self.data		= data
		self.tensors		= tensors
		self.STEP		= data.getTParam()[0]
		name = 'Machine_Learning.Training.'
		if self.DT == 'MNIST' or self.DT == 'CIFAR10':
			self.cname = 'BaseTrainer'
			self.tname = name + self.cname
			self.info = [self.data.getB()[0],
					self.data.getB()[1],self.tensors.getTensors()[0]]
		elif self.DT == 'PSINGLE':
			self.cname = 'ProcTrainer'
			self.tname = name + self.cname
			#Only run getP ONCE else it BORKS
			[a,b] = self.data.getP()
			self.info = [a,b,self.data.getI()[0]]
		else:
			quit()
		
		try:
			self.tr = i.import_module(self.tname)
		except ImportError:
			sys.exit('Failed to import, poor trainer name: ' + self.tname)
	#Dynamically run training based on prev class	
	def RunTraining(self):
		a = [self.info[0],self.info[1],self.info[2],
		self.tensors.getTensors()[1],self.tensors.getTensors()[2],
		self.tensors.getTensors()[5],self.tensors.getTensors()[3],
		self.tensors.getTensors()[4]]
		trclass = getattr(self.tr,self.cname)
		Train = trclass(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
		Train.training(self.DT,self.STEP,self.GT,self.LOG)
	


























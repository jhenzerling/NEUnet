#A support module to set up the various tensors before training
#Machine Learning Handling
from Machine_Learning import Cost as cost
from Logging import mod_graph as graph
#Tensorflow Libraries
import tensorflow as tf
import importlib as i
#Main Configuration
from Configs import config_main as config

#Class to set tensors and other information
class Setup:
	#Set Primary Tensors For Use in Code
	def __init__(self,data,datatrigger):
		if datatrigger == 'MNIST' or datatrigger == 'CIFAR10':
			[hsize,vsize,colours,cnumb] = data.getDParam()
			with tf.variable_scope('Input_Tensor'):
				self.x 		= tf.placeholder(tf.float32, [None,hsize,vsize,colours], name = 'input')
			with tf.variable_scope('Result'):
				self.y_ 	= tf.placeholder(tf.float32, shape=[None, cnumb], name = 'result')
			with tf.variable_scope('Keep_Prob'):			
				self.keep_prob 	= tf.placeholder(tf.float32, name = 'keep_prob')
		elif datatrigger == 'PSINGLE':
			with tf.variable_scope('Input_Tensor'):
				self.x 		= data.getI()[2]
			with tf.variable_scope('Result'):
				self.y_		= data.getI()[1]
			with tf.variable_scope('Keep_Prob'):
            			self.keep_prob 	= tf.placeholder(tf.float32, name = 'keep_prob')
		else:
			quit()
	#Pick and do Network Setup after Import
	def setArchitecture(self,archi,cn,tt,dt):
		self.arname = 'Machine_Learning.Networks.' + archi
		print('Using ' + archi)
		try:
			self.ar = i.import_module(self.arname)
		except ImportError:
			sys.exit('Failed to import, poor architecture name: ' + archi)
		if archi == 'u-resnet':
			quit()
		else:
			self.net,self.end_points = self.ar.build(self.x,cn,tt,dt)
			self.setCost()
			return [self.net,self.end_points]
	#Set Training Variables That Depend on Network
	def setCost(self):
		learnrate = config.LEARN['Rate']
        	[self.costfunction,
		self.train_step,
		self.accuracy] = cost.PSingle_Cost_Setup(self.y_,self.net,learnrate)
			
	#Raw Getter		
	def getTensors(self):
		return [self.x,self.y_,self.keep_prob,self.costfunction,
				self.train_step,self.accuracy]

#Technically this should be wrapped up in the trainer imo


#Pipeline should be:

#1. Infor allocation -> MultiOutput
#2. Tensor/Config/Proc Setup -> Takes in all from previous
#3. Training and Output -> Separate objects for each dataset


#Should allow the main to be UNCHANGED forevermore by wrapping up stuff in sep. modules

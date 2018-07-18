#Class to call in data with ease
import tensorflow as tf
import importlib as i
import sys


class data:

	#Instantiation of an object -> dataset is a string which is data NAME
	def __init__(self,dataset):
		print('Utilizing ' + dataset + ' Data')
		self.modname = 'mod_' + dataset
		self.impmod = 'Loaders.' + self.modname
		try:
			self.mod = i.import_module(self.impmod)
		except ImportError:
			sys.exit('Failed to import, poor data trigger name: ' + dataset)
	
	#Variable Getters - Inherited from Loaders - LETS REMOVE THE NEED FOR THEM IN LOADERS
	#All Loaders should have hsize,vsize,colours,cnumb,batchsize,stepnumber,trainsize,testsize,
	#This uniformity leads to below
	def getDParam(self):
		return [self.mod.hsize,self.mod.vsize,self.mod.colours,self.mod.cnumb]
	def getTParam(self):
		return [self.mod.stepnumber,self.mod.batchsize,self.mod.testsize,self.mod.trainsize]
	def getB(self):
		return [self.mod.trainbatch,self.mod.testbatch]
	def getP(self):
		self.mod.trproc = self.mod.IOPrep('train',self.mod.batchsize)
    		self.mod.teproc = self.mod.IOPrep('test',self.mod.batchsize)
    		return [self.mod.trproc,self.mod.teproc]
	def getI(self):
		self.mod.Itensor = tf.placeholder(
			tf.float32,[None,self.mod.hsize*self.mod.vsize*self.mod.colours], name='input')
    		self.mod.Ltensor = tf.placeholder(
			tf.float32,[None,self.mod.cnumb], name='label')
    		self.mod.Itensor2D = tf.reshape(
			self.mod.Itensor, 
			[-1,self.mod.hsize,self.mod.vsize,self.mod.colours], name='2D_input')
    		return [self.mod.Itensor,self.mod.Ltensor,self.mod.Itensor2D]
	def getCFG(self,name):
    		if(name == 'train'):
        		self.cfg = self.mod.Train_cfg
   		elif(name == 'test'):
        		self.cfg = self.mod.Test_cfg
    		else:
        		print('Bad name, check CFG')
    		return self.cfg

	
	#Additional Functionality -> This class should allow VIEWING data
	#Want to show by EVENT, by TYPE, plot, TRUTH INFO, etc
	#dataclass should generically allow access of how everything looks
	#The analysis class later on will only allow looking at the softmaxes/outputs nawmean
	#DATA -> IMAGES,RAW,REFER TO LOADER
	#LOADER -> GENERALIZE THE DATA ACCESS -> PREPROCESSING MODULES IF NEEDED
	#ANALYSIS -> OUTPUTS,SOFTMAXES,

	#DATAOBJECT calls from modules to get its output in order to decrease complexity
	#on the network end of the program

	#DO NOT NEED STATISTICS SINCE IT COMES IN THE ANALYSIS

	#Truth Information

	#Plotters

	#The input-output info will be in the analysis framework

#Network Training Module

#################################################################################################################

#LIBRARIES AND BACKGROUND

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
#Machine Learning Handling
from Machine_Learning import Learning as learn
from Machine_Learning import TensorSetup as TS
from Machine_Learning.Training import Trainer as TR
#Tensorflow Libraries
import tensorflow as tf
#Main Configuration
from Configs import config_main as cm

tf.reset_default_graph()

#################################################################################################################

#SETTINGS AND EDITING

#Network ON/OFF - Useful for Editing
trigger 	= cm.MAIN['Trigger']
archi 		= cm.MAIN['Archi']
#Data Selection
datatrigger 	= cm.MAIN['Data']
#Properties to tune runs of neunet.py
debugtrigger 	= cm.MAIN['Debug']
traintrigger 	= cm.MAIN['Train']
plottrigger 	= cm.MAIN['Plot']
graphtrigger 	= cm.MAIN['Graph']
loggingnumber 	= cm.MAIN['Logging']
#Learning and Saving Rate
learnrate 	= cm.LEARN['Rate']
saverate 	= cm.LEARN['Save']

#################################################################################################################
#DATA SELECTION

#Load Data 
data = d.data(datatrigger)
#In Variable scope, set the variables
with tf.variable_scope(datatrigger):
	[hsize,vsize,colours,cnumb] = data.getDParam()
    	[stepnumber,batchsize,testsize,trainsize] = data.getTParam()
	#Check if it's a batch program (MNIST/CIFAR10), if not, find procs (PSINGLE etc)
	try:
    		[trainbatch, testbatch] = data.getB()
	except AttributeError:
		#[trproc, teproc] = data.getP()
		[rawinput,labinput,input2d] = data.getI()

#PLOTTING BUSTED FOR NOW, WILL COME BACK TO IT LATER -> REORGANIZE FILES
if(plottrigger == True):
	number = int(raw_input('Enter Image Number to Plot: '))
	log.CIFAR10_plot(testbatch,hsize,vsize,number)


###########################################################################################################

#Set the Tensors Used in the Network if Needed
if trigger == True:
    #Script unit test
    if __name__ == '__main__':
	Tensors 		= TS.Setup(data,datatrigger)
	[net,end_points] 	= Tensors.setArchitecture(archi,cnumb,traintrigger,debugtrigger)
	[x,y_,keep_prob,
	costfunction,train_step,
	accuracy] 		= Tensors.getTensors()

	#Histograms for Tensorboard
    	graph.setHistograms('my_summ')

    	#SETTING UP TENSORBOARD GRAPH
   	if graphtrigger == True:
        	print('Producing TensorBoard Graph.')

###########################################################################################################

#Perform the Training of the Network

    MLT = TR.Trainer(data,Tensors)
    MLT.RunTraining()

#End Code
else:
    print('Trigger not ON. Terminating')
    end = time.time()
    print('Process Time: ' + str(end - start) + ' seconds')
    quit()

###########################################################################################################

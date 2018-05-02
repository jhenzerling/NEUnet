#PSingle HANDLING
from __future__ import print_function

import numpy as np
import tensorflow as tf
import ROOT

#from ROOT import TChain
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

from matplotlib import pyplot as plt
import time

from Logging import mod_log as log
from Configs import config_main as config

larcv.load_pyutil()

#File use and paths
dpath = '/user/jhenzerling/work/neunet/Data/PSingle/'
trainfile = 'train_50k.root'
testfile = 'test_40k.root'
trpath = dpath + trainfile
tepath = dpath + testfile

#File sizes
testsize = config.DATA['Test']
trainsize = config.DATA['Train']

#Image information
hsize = config.DATA['H']
vsize = config.DATA['V']
colours = config.DATA['Colours']
cnumb = config.DATA['Classes']

#Axes are 0=xy,1=yz,2=zx
axes = config.DATA['Axes']

#Machine Learning
stepnumber = config.LEARN['Step']
batchsize = config.LEARN['Batch']

#Collect the Config File
Train_cfg = {'filler_name':'TrainCFG',
             'verbosity':0,
             'filler_cfg':'Configs/TrainCFG.cfg'}

Test_cfg = {'filler_name':'TestCFG',
            'verbosity':0,
            'filler_cfg':'Configs/TestCFG.cfg'}
        
##################################################################
#PSingle Methods

#Construct and prepare memory for the threadio
def IOPrep(name,b):
    if(name == 'train'):
        cfg = Train_cfg
    elif(name == 'test'):
        cfg = Test_cfg
    else:
        print('Bad name, check CFG')
    
    proc = larcv_threadio()
    proc.configure(cfg)
    proc.start_manager(b)
    #Need sleep for manager to finish loading
    time.sleep(2)
    proc.next()

    return proc

#Fetch data and pull out the image and its label
def allocate(name,io):
    im = name + '_image'
    la = name + '_label'

    #Batched pydata
    data = io.fetch_data(im)
    labe = io.fetch_data(la)

    #Nonetype, (list of numpys)
    image = data.data()
    label = labe.data()

    return [image,label]

#Convert the flat data into a 2D image for use in the network
def formatTensors(da):
    #da = [im,lab]
    image_tensor = tf.convert_to_tensor(da[0])
    image_tensor_2d = tf.reshape(image_tensor, [batchsize, hsize, vsize, colours])
    label_tensor = tf.convert_to_tensor(da[1])
    
    return [image_tensor_2d, label_tensor]

#Convert PDG's to particle name
def OHtoName(onehot):
    name = ''
    if(onehot[0]==1):
        name = 'Electron'
    elif(onehot[1]==1):
        name = 'Muon'
    elif(onehot[2]==1):
        name = 'Gamma'
    elif(onehot[3]==1):
        name = 'Pion'
    elif(onehot[4]==1):
        name = 'Proton'
    print(name)
    return name


#########################################################################################
#Getters

#Create Placeholder for use in main (B,256,256,1)
def get_Input():
    Itensor = tf.placeholder(tf.float32,[None,hsize*vsize*colours], name='input')
    Ltensor = tf.placeholder(tf.float32,[None,cnumb], name='label')
    Itensor2D = tf.reshape(Itensor, [-1,hsize,vsize,colours], name='2D_input')
    return [Itensor,Ltensor,Itensor2D]

#Get the data parameters
def get_dataparams():
    return [hsize,vsize,colours,cnumb]

#Get the training parameters
def get_trainparams():
    return [stepnumber,batchsize,testsize,trainsize]

#Get batches
def get_batch():
    return [trainbatch,testbatch]

#Set and retrieve IO's
def get_proc():
    trproc = IOPrep('train',batchsize)
    teproc = IOPrep('test',batchsize)
    return [trproc,teproc]

#Getter for the CFG
def get_CFG(name):
    if(name == 'train'):
        cfg = Train_cfg
    elif(name == 'test'):
        cfg = Test_cfg
    else:
        print('Bad name, check CFG')

    return cfg












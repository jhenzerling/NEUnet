#PSingle HANDLING
from __future__ import print_function

import numpy as np
import tensorflow as tf

import ROOT as R
import os,time

from larcv import larcv
from larcv.dataloader2 import larcv_threadio

from Configs import config_main as cm
from Configs import config_data as cd

larcv.load_pyutil()

path 		= cd.PSINGLE['Path']
#Check for data
if(os.listdir(path) != []):
    print('PSINGLE Data Present.')
else:
    print('PSINGLE Data Not Present.')
    quit()

#Files and Paths
trf 		= cd.PSINGLE['TRF']
tef 		= cd.PSINGLE['TEF']
trpath 		= path + trf
tepath 		= path + tef
#Image information
hsize 		= cd.PSINGLE['H']
vsize 		= cd.PSINGLE['V']
colours 	= cd.PSINGLE['Colours']
cnumb 		= cd.PSINGLE['Classes']
axes 		= cd.PSINGLE['Axes'] #Axes are 0=xy,1=yz,2=zx
#File sizes
trainsize 	= cd.PSINGLE['Train']
testsize 	= cd.PSINGLE['Test']
#Machine Learning
stepnumber 	= cm.LEARN['Step']
batchsize 	= cm.LEARN['Batch']
#Collect the Config File
Train_cfg 	= cd.PSINGLE['TRCFG']
Test_cfg 	= cd.PSINGLE['TECFG']   
     
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

#Convert the flat data into a 2D image for use in the network
def formatTensors(da):
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
        name = 'Photon'
    elif(onehot[3]==1):
        name = 'Pion'
    elif(onehot[4]==1):
        name = 'Proton'
    print(name)
    return name

#########################################################################################
#Gives range of energies and amount of each pdg from get-go, this is an example for further data
def spread():
	#0elec,1gamm,2muo,3pio,4prot
	PDG2NAME = {11   : 0,
                    22   : 1,
                    13   : 2,
                    211  : 3,
                    2212 : 4}	
	bnumb = 20
	chain = R.TChain("particle_mctruth_tree")
	chain.AddFile(trpath)
	pdgbin = np.zeros(cnumb)
	ebin = np.zeros(chain.GetEntries())
	#Store pdg's and init energies
	for x in range(chain.GetEntries()):
		chain.GetEntry(x)
		truth = chain.particle_mctruth_branch
		parray = truth.as_vector()
		pdgbin[PDG2NAME[parray[0].pdg_code()]] += 1
		ebin[x] = parray[0].energy_init()*1000 - larcv.ParticleMass(parray[0].pdg_code())
	#Sort the array and find the range of TOTAL energies (kin+mass)
	ebin2 = np.sort(ebin)
	low = ebin2[0]
	high = ebin2[-1]
	dist = (high - low)/bnumb
	ebin3 = np.zeros(bnumb)
	for x in range(len(ebin)):
		for y in range(bnumb):
			if (low + y*dist <= ebin2[x] <= low + (y+1)*dist):
			  	ebin3[y] += 1
	return[pdgbin,ebin3]













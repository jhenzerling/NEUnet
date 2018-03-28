#PSingle HANDLING

#DO THIS AND ALSO FIGURE OUT TENSORBOARD IF CAN 

from __future__ import print_function

import numpy as np
import tensorflow as tf
import ROOT
from ROOT import TChain
from larcv import larcv

from Logging import mod_log as log
from matplotlib import pyplot as plt


#NEED TO CREATE LOADER FOR PSINGLE, EXTRACT FROM ROOT ETC


#DATA LOADER

#DATA FORMATTER (OPTIONAL) AND OTHER PREPROCESSING

#BATCHING

#GETTERS


ROOT.TFile.Open('test_40k.root').ls()
chain_image2d = ROOT.TChain('image2d_data_tree')
chain_image2d.AddFile('test_40k.root')
print(chain_image2d.GetEntries(),'entries found!')

# Get a specific event (first entry)
chain_image2d.GetEntry(0)
cpp_object = chain_image2d.image2d_data_branch
print('Object type:',cpp_object)

# Get std::vector<larcv::Image2D>
image2d_array = cpp_object.as_vector()
# Dump images
fig, axes = plt.subplots(1, image2d_array.size(), figsize=(12,4), facecolor='w')
for index,image2d in enumerate(image2d_array):
    image2d_numpy = larcv.as_ndarray(image2d)
    axes[index].imshow(image2d_numpy, interpolation='none',cmap='jet')
    # Find bounds for non-zero pixels + padding of 5 pixels
    nz_pixels=np.where(image2d_numpy>0.0)
    ylim = (np.min(nz_pixels[0])-5,np.max(nz_pixels[0])+5)
    xlim = (np.min(nz_pixels[1])-5,np.max(nz_pixels[1])+5)
    # Adjust for allowed image range
    ylim = (np.max((ylim[0],0)), np.min((ylim[1],image2d_numpy.shape[1]-1)))
    xlim = (np.max((xlim[0],0)), np.min((xlim[1],image2d_numpy.shape[0]-1)))
    # Set range
    axes[index].set_ylim(ylim)
    axes[index].set_xlim(xlim)
plt.show()

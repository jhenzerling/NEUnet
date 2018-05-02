# neunet
Neunet is a convolutional network used to process images, eventually hoping to be used to analyze neutrino interactions in SBND. The network uses Tensorflow and has TensorBoard integration, along with modularity allowing further datasets to be used. 
Data as of now needs to be self-imported, along with any libraries or modules. The Network will eventually require LArCV and ROOT for neutrino analysis.

In order to run the network, simply type 'python main.py' and it will run. Settings are changed in the 'main.py' file, and any dataset-specific settings are changed in their respective loader file in '/Loaders/'.

I include an inference package via 'analysis.py' which runs separately from 'main.py'. This can check saved weights and help look at data. As of May only works for PSingle.

This neural network requires numpy, ipython, tensorflow_gpu, ROOT, and LArCV. The network was ran using a virtual environment with these modules along with jupyter. Full setup instructions will be included in a later patch.


Results (Last Checked - 2 May)
MNIST - 95%
CIFAR10 - 70%
PSingle - 80%

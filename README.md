# neunet
Neunet is a convolutional network used to process images, eventually hoping to be used to analyze neutrino interactions in SBND. The network uses Tensorflow and has TensorBoard integration, along with modularity allowing further datasets to be used. As of March '18, we have seen 95% on MNIST and 70% on CIFAR10.
Data as of now needs to be self-imported, along with any libraries or modules. The Network will eventually require LArCV and ROOT for neutrino analysis.
In order to run the network, simply type 'python main.py' and it will run. Settings are changed in the 'main.py' file, and any dataset-specific settings are changed in their respective loader file in '/Loaders/'.

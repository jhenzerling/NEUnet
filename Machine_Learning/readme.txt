This is where we store the neural network and other such modules.

'Network.py' houses the CNN architecture as a callable function. Changing the architecture or adding substructures should be done here.

'Cost.py'  holds the learning functions, such as cost and accuracy. The learning rate is called in the main, so no settings-changes need to be done here.

'Learning.py' holds information for dealing with and setting the feeds in the main.

This contains the various data-loading files to use in the neural network.

The various 'mod_DATASET' files house the preprocessing and formatting of the data into an architecture-usable way. Each module houses loaders to import the data, a variety of tools to preprocess and batch, and getters to provide the main a uniform way to get each dataset. For adding other datasets, make a new 'mod_DATASET' and use that to bring in the data to 'main.py'.

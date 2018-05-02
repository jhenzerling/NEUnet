#Configuration for main.py
MAIN = {
    'Data'   : 'PSingle',
    'Archi'  : 'B2',
    'Trigger': True,
    'Debug'  : True,
    'Train'  : True,
    'Plot'   : False,
    'Graph'  : True,
    'Logging': 250,
    'Save'   : 100
}

#Configuration for main network
NETWORK = {
    'Network': 'Neunet',
    'Filters': 32,
    'Kernal' : [3,3],
    'Stride' : 1,
    'FCSize' : 1024,
    'CLNumb' : 7,
    'FCNumb' : 2,
    'PoolStride': 2
}

#Configuration for Data
DATA = {
    'Train'    : 50000,
    'Test'    : 40000,
    'Colours' : 1,
    'Classes' : 5,
    'Axes'    : 3,
    'H'       : 256,
    'V'       : 256
}

#Configuration for Learning
LEARN = {
    'Rate'  : 0.00005,
    'Step'  : 5000,
    'Batch' : 50
}
    

#Configuration for main.py
MAIN = {
    'Data'   : 'MNIST',
    'Archi'  : 'B1net',
    'Trigger': True,
    'Debug'  : True,
    'Train'  : True,
    'Plot'   : False,
    'Graph'  : True,
    'Logging': 100,
    
}

#Configuration for main network
NETWORK = {
    'Network': 'Neunet',
    'Filters': 32,
    'Kernal' : [3,3],
    'Stride' : 1,
    'FCSize' : 512,
    'CLNumb' : 2,
    'FCNumb' : 2,
    'PoolStride': 2
}

#Configuration for Learning
LEARN = {
    'Rate'  : 0.001,
    'Step'  : 1000,
    'Batch' : 50,
    'Save'  : 100
}
    

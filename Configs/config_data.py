#Configuration for Datasets
MNIST = {
    'Train'    : 55000,
    'Test'    : 10000,
    'Colours' : 1,
    'Classes' : 10,
    'Axes'    : 1,
    'H'       : 28,
    'V'       : 28,
    'Path'    : '/user/jhenzerling/work/neunet/Data/MNIST/'
}

CIFAR10 = {
    'Train'    : 50000,
    'Test'    : 10000,
    'Colours' : 3,
    'Classes' : 10,
    'Axes'    : 1,
    'H'       : 32,
    'V'       : 32,
    'Path'    : '/user/jhenzerling/work/neunet/Data/CIFAR10/'
}

PSINGLE = {
    'Train'    : 50000,
    'Test'    : 40000,
    'Colours' : 1,
    'Classes' : 5,
    'Axes'    : 3,
    'H'       : 256,
    'V'       : 256,
    'Path'    : '/user/jhenzerling/work/neunet/Data/PSINGLE/',
    'TRF'     : 'train_50k.root',
    'TRCFG'   : {'filler_name':'TrainCFG','verbosity':0,'filler_cfg':'Configs/TrainCFG.cfg'},
    'TEF'     : 'test_40k.root',
    'TECFG'   : {'filler_name':'TestCFG','verbosity':0,'filler_cfg':'Configs/TestCFG.cfg'}
}

PMULTI = {
    'Train'    : 8500,
    'Test'    : 9000,
    'Colours' : 1,
    'Classes' : 5,
    'Axes'    : 3,
    'H'       : 1280,
    'V'       : 1986,
    'V2'      : 1666,
    'Path'    : '/hepstore/jhenzerling/sbnd_dl_samples/',
    'TRF'     : 'train_50k.root',
    'TRCFG'   : {'filler_name':'TrainCFG','verbosity':0,'filler_cfg':'Configs/TrainCFG.cfg'},
    'TEF'     : 'test_40k.root',
    'TECFG'   : {'filler_name':'TestCFG','verbosity':0,'filler_cfg':'Configs/TestCFG.cfg'}
}

SEMSEGTUT = {
    'Train'   : 15000,
    'Test'    : 10000,
    'Colours' : 1,
    'Classes' : 3,
    'Axes'    : 3,
    'H'       : 256,
    'V'       : 256,
    'V2'      : 1666,
    'Path'    : '/user/jhenzerling/work/neunet/Data/SEMSEGTUT/',
    'TRF'     : 'train_15k.root',
    'TRCFG'   : {'filler_name':'TrainCFGSEG','verbosity':0,'filler_cfg':'Configs/TrainCFGSEG.cfg'},
    'TEF'     : 'test_10k.root',
    'TECFG'   : {'filler_name':'TestCFGSEG','verbosity':0,'filler_cfg':'Configs/TestCFGSEG.cfg'}
}


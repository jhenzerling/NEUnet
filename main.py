#Network Training Module

#################################################################################################################

#LIBRARIES AND BACKGROUND

#Numpy for number-handling
import numpy as np

#File Handling
from Logging import mod_log as log
from Logging import mod_graph as graph
import sys,os,time,datetime
#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from tensorflow.python.client import device_lib
#print device_lib.list_local_devices()
#Start Clock for timing
start = time.time()

#Machine Learning Handling
from Machine_Learning import Cost as cost
from Machine_Learning import Network as network
from Machine_Learning import Learning as learn

#Tensorflow Libraries
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#LARCV
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

#Main Configuration
from Configs import config_main as config

tf.reset_default_graph()

#################################################################################################################

#SETTINGS AND EDITING

#Network ON/OFF - Useful for Editing
trigger = config.MAIN['Trigger']
archi = config.MAIN['Archi']
#Data Selection
datatrigger = config.MAIN['Data']
debugtrigger = config.MAIN['Debug']
traintrigger = config.MAIN['Train']
plottrigger = config.MAIN['Plot']
graphtrigger = config.MAIN['Graph']
loggingnumber = config.MAIN['Logging']

#Numbers
learnrate = config.LEARN['Rate']
saverate = config.MAIN['Save']

#################################################################################################################

#DATA SELECTION

if datatrigger == 'CIFAR10':
    print('Utilizing CIFAR10 Data.')
    from Loaders import mod_cifar10
    with tf.variable_scope('Cifar10'):
        [hsize,vsize,colours,cnumb] = mod_cifar10.get_dataparams()
        [stepnumber,batchsize,testsize] = mod_cifar10.get_trainparams()
        [trainbatch,testbatch] = mod_cifar10.get_batch()

    if(plottrigger == True):
        number = int(raw_input('Enter Image Number to Plot: '))
        log.CIFAR10_plot(testbatch,hsize,vsize,number)
        
    #################################################################################################################

elif datatrigger == 'MNIST':
    print('Utilizing MNIST Data.')
    from Loaders import mod_mnist
    with tf.variable_scope('MNIST'):
        [hsize,vsize,colours,cnumb] = mod_mnist.get_dataparams()
        [stepnumber,batchsize,testsize] = mod_mnist.get_trainparams()
        [trainbatch, testbatch] = mod_mnist.get_batch()

    if(plottrigger == True):
        number = int(raw_input('Enter Image Number to Plot: '))
        log.MNIST_plot(trainbatch,hsize,vsize,number)

    #################################################################################################################

elif datatrigger == 'PSingle':
    #In this module the batching is done differently so we need extra steps here
    print('Utilizing PSingle Data.')
    from Loaders import mod_psingle
    with tf.variable_scope('PSingle'):
        [hsize,vsize,colours,cnumb] = mod_psingle.get_dataparams()
        [stepnumber,batchsize,testsize,trainsize] = mod_psingle.get_trainparams()
        [trproc,teproc] = mod_psingle.get_proc()
        [rawinput,labinput,input2d] = mod_psingle.get_Input()

    if(plottrigger == True):
        number = int(raw_input('Enter Image Number to Plot: '))
        log.PSingle_plot(trainbatch,hsize,vsize,number)

    #################################################################################################################

#ERROR CATCHING
else:
    print('Incorrect Data Selection - Terminating')
    quit()

###########################################################################################################

#RUNNING THE SESSION

if trigger == True:

    #Script unit test
    if __name__ == '__main__':
    
        #Creating placeholder in shape of input image
        with tf.variable_scope('Input_Tensor'):
            if datatrigger == 'PSingle':
                x = input2d
            else:
                x = tf.placeholder(tf.float32, [None,hsize,vsize,colours], name = 'input')
        if archi == 'B1':
            print('Using Architecture 1')
            net,end_points = network.build(x,cnumb,traintrigger,debugtrigger)
        elif archi == 'B2':
            print('Using Architecture 2')
            net,end_points = network.build2(x,cnumb,traintrigger,debugtrigger)
        else:
            print('Invalid Architecture')

        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        #Create Placeholder of possible results
        with tf.variable_scope('Result'):
            if datatrigger == 'PSingle':
                y_ = labinput
            else:
                y_ = tf.placeholder(tf.float32, shape=[None, cnumb], name = 'result')
        with tf.variable_scope('Keep_Prob'):
            keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

###########################################################################################################

#COST FUNCTION AND TRAINING STEP SUMMARY

    #Set cost function and accuracy
    with tf.variable_scope('CostFunct'):
        [costfunction,train_step,accuracy] = cost.PSingle_Cost_Setup(y_,net,learnrate)
        tf.summary.scalar("Cost Function", costfunction)
        tf.summary.scalar("Accuracy", accuracy)
        
    if datatrigger == 'PSingle':
        with tf.variable_scope('Save_Images'):
            tf.summary.image('train_input',mod_psingle.formatTensors(mod_psingle.allocate('train',trproc))[0],10)

    #Histograms for Tensorboard
    graph.setHistograms('my_summ')

    #SETTING UP TENSORBOARD GRAPH
    if graphtrigger == True:
        print('Producing TensorBoard Graph.')
   
###########################################################################################################

    #Running the session with configuration as follows:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        #Initialize the nodes and variables
        with tf.variable_scope('Session'):
            init = tf.global_variables_initializer()
            sess.run(init)
        #Log the graph
        if graphtrigger == True:
            graph.TBPrint()
            [writertrain,writertest,merged] = graph.setSummaries(sess)
            saver = tf.train.Saver()
        #Using batches from dataset to train
        for i in range(stepnumber):
            if datatrigger == 'PSingle':
                trproc.next()
                trainfeed = learn.PSingle_Feeds(trproc,rawinput,y_,'train')
            elif datatrigger == 'CIFAR10':
                [trainfeed,testfeed] = learn.CIFAR10_Feeds(trainbatch[0],trainbatch[1],x,y_)
            elif datatrigger == 'MNIST':
                [trainfeed,testfeed] = learn.MNIST_Feeds(trainbatch[0],trainbatch[1],x,y_)
            else:
                print('Bad Train and/or Test Feeds')
                    
            #Logging at certain steps
            with tf.variable_scope('Feeding'):
                if i % (stepnumber/(loggingnumber)) == 0:
                    if datatrigger == 'PSingle':
                        teproc.next()
                        testfeed = learn.PSingle_Feeds(teproc,rawinput,y_,'test')

                    #Graph Operations
                    if graphtrigger == True:
                        [summarytrain,acctrain,summarytest,acctest] = graph.SummFeeder(merged,
                                                                    accuracy,sess,trainfeed,testfeed)
                        graph.SummAdder(writertrain,writertest,summarytrain,summarytest,i)
                    train_accuracy = accuracy.eval(feed_dict=trainfeed)
                    test_accuracy = accuracy.eval(feed_dict=testfeed)
                    train_loss = costfunction.eval(feed_dict=trainfeed)
                    test_loss = costfunction.eval(feed_dict=testfeed)
                    log.outputprint(stepnumber,train_accuracy,test_accuracy,
                                    train_loss,test_loss,start - time.time(),i)
                   
            #Training Step
            with tf.variable_scope('Training'):
                train_step.run(feed_dict=trainfeed)
        
            #Log Weights 2x as often as Logging
            if i % saverate == 0:
                ssf_path = saver.save(sess, 'Weights/neunet',global_step=i)

        #Close out the Threads
        coord.request_stop()
        coord.join(threads)
        if graphtrigger == True:
            graph.WCloser(writertrain,writertest)

else:
    print('Trigger not ON. Terminating')
    end = time.time()
    print('Process Time: ' + str(end - start) + ' seconds')
    quit()

###########################################################################################################

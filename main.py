#################################################################################################################

#LIBRARIES AND BACKGROUND

#Numpy for number-handling
import numpy as np

#File Handling
from Logging import mod_log as log
#from matplotlib import pyplot as plt
import sys
import time
import datetime
time.clock()

#Machine Learning Handling
from Machine_Learning import Cost as cost
from Machine_Learning import Network as network

#Tensorflow Libraries
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

tf.reset_default_graph()

#################################################################################################################

#SETTINGS AND EDITING

#Network ON/OFF - Useful for Editing
trigger = False
#Data Selection
datatrigger = 'PSingle'
debugtrigger = True
traintrigger = True
plottrigger = False
graphtrigger = False

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
    print('Utilizing PSingle Data.')
    from Loaders import mod_psingle
    with tf.variable_scope('PSingle'):
        [hsize,vsize,colours,cnumb] = mod_psingle.get_dataparams()
        #[stepnumber,batchsize,testsize] = mod_psingle.get_trainparams()
        #[trainbatch, testbatch] = mod_psingle.get_batch()

    if(plottrigger == True):
        number = int(raw_input('Enter Image Number to Plot: '))
        #log.PSingle_plot(trainbatch,hsize,vsize,number)

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
            x = tf.placeholder(tf.float32, [None,hsize,vsize,colours])
        net,end_points = network.build(x,cnumb,traintrigger,debugtrigger)
        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        #Create Placeholder of possible results
        with tf.variable_scope('Result'):
            y_ = tf.placeholder(tf.float32, shape=[None, cnumb])
        with tf.variable_scope('Keep_Prob'):
            keep_prob = tf.placeholder(tf.float32)

###########################################################################################################

#COST FUNCTION AND TRAINING STEP SUMMARY

    #Set cost function and accuracy
    with tf.variable_scope('CostFunct'):
        [costfunction,train_step,accuracy] = cost.MNIST_Cost_Setup(y_,net,1e-5)
        tf.summary.scalar("Cost Function", costfunction)
        tf.summary.scalar("Accuracy", accuracy)

    #Histograms for Tensorboard
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name,var,collections=["my_summ"])

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
            tf.trainable_variables(tf.GraphKeys.GLOBAL_VARIABLES)
            merged = tf.summary.merge_all()
            print('CALL THE FOLLOWING COMMAND FOR TENSORBOARD: ')
            print('tensorboard --logdir=run1:/user/jhenzerling/work/neunet/Graphs/ --port 8008')
            writer=tf.summary.FileWriter('Graphs', sess.graph)

        #Put batches into the same queue
        Qtrainbatch0,Qtrainbatch1 = sess.run([trainbatch[0],trainbatch[1]])
        trainfeed = {x: Qtrainbatch0, y_: Qtrainbatch1, keep_prob: 0.8}

        #Using batches from dataset to train
        for i in range(stepnumber):

            #Graph Operations
            if graphtrigger == True:
                summary,acc = sess.run([merged,accuracy], feed_dict=trainfeed)
                writer.add_summary(summary,i)

            #Logging at certain steps
            with tf.variable_scope('Feeding'):
                if i % (stepnumber/(10)) == 0:
                    train_accuracy = accuracy.eval(feed_dict=trainfeed)
                    log.output_print(stepnumber,train_accuracy,time.clock(),i)
                   
            #Training Step
            with tf.variable_scope('Training'):
                train_step.run(feed_dict=trainfeed)

        #Print the output accuracy
        #Input batches into the same queue
        Qtestbatch0,Qtestbatch1 = sess.run([testbatch[0],testbatch[1]])
        testfeed = {x: Qtestbatch0, y_: Qtestbatch1, keep_prob: 1.0}
        print('Test Accuracy: (%g)\t' % accuracy.eval(feed_dict=testfeed))
        print('Process Time: (' + str(time.clock()) + ') seconds\t')

        #Close out the Threads
        coord.request_stop()
        coord.join(threads)
        writer.close()

else:
    print('Trigger not ON. Terminating')
    print('Process Time: ' + str(time.clock()) + ' seconds')
    quit()

###########################################################################################################

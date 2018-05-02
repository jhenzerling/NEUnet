#Module for Recording and Logging

import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from larcv import larcv
import ROOT
from Configs import config_main as config

#MNIST plotter to check inputs
def MNIST_plot(trainbatch,hsize,vsize,number):
    #PLOTTING#
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    #print(np.argmax(trainbatch[0][number]))
    img_data = (trainbatch[0][number].eval(session = sess)).reshape([hsize,vsize])
    plt.imshow(img_data)
    plt.show()
    print('Which has class: ' + str(trainbatch[1][number].eval(session=sess)))

#Main difference compared to MNISt is requirement of queues to deal with batches
def CIFAR10_plot(trainbatch,hsize,vsize,number):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)

        #Initialize the nodes and variables
        sess.run(tf.global_variables_initializer())
        img,lab = sess.run([trainbatch[0][number],trainbatch[1][number]])
        plt.imshow(img)
        plt.show()
        print('Which has class: ' + str(lab))
        coord.request_stop()
        coord.join(threads)

#Plot a single PSingle Event
def PSingle_plot(image,label,dim,number):
    image_reshaped = image.reshape(dim[:-1])
    fig,ax = plt.subplots(figsize=(8,8))
    plt.imshow(image_reshaped[number],cmap='jet',interpolation='none')
    plt.show()

##################################################################################################

#Output training data to screen
def outputprint(step,tracc,teacc,trloss,teloss,time,i):
    print('Progress: (%g) %%, \t\t\t Time: (%d)' % (i*100/step, time))
    print('Training Accuracy: (%g), \t\t\t Training Loss: (%g)' % (tracc, trloss))
    print('Testing Accuracy: (%g), \t\t\t Testing Loss: (%g)' % (teacc, teloss))

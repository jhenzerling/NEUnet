#This module is for building networks separately from the main
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf
from Configs import config_main as config

#BUILDING THE NETWORK

def build(input_tensor, num_class, trainable, debug):

    outputs_collections = "neunet"

    #Take in input data as 'net' object
    #Expect an input of [BatchSize, Channels, HSize, VSize]
    net = input_tensor
    if debug: print 'Input Tensor: ', input_tensor.shape

    #Define the number of filters to convolve
    filters = config.NETWORK['Filters']
    #Define Kernel size and stride
    kernsize = config.NETWORK['Kernal']
    strider = config.NETWORK['Stride']
    poolstride = config.NETWORK['PoolStride']
    #Define Fully Connected Size
    fcsize = config.NETWORK['FCSize']
    #Define Number of Conv and FC Layers
    clnumb = config.NETWORK['CLNumb']
    fcnumb = config.NETWORK['FCNumb']

    #Loop to set hidden layers in the network
    with tf.variable_scope('Network'):
        with tf.variable_scope('Deep_Conv_Layers'):
            for step in xrange(clnumb):
                #Convolve the network via slim
                net = slim.conv2d(inputs      = net,       # input tensor
                                  num_outputs = filters,   # number of filters/feature maps
                                  kernel_size = kernsize,     # kernel size
                                  stride      = strider,         # stride size
                                  trainable   = trainable, # train or inference
                                  activation_fn = tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer = tf.zeros_initializer(),
                                  scope       = 'Conv_Layer_%d' % step)
                
                #Max Pool the network
                net = slim.max_pool2d(inputs      = net,    # input tensor
                                      kernel_size = kernsize,  # kernel size
                                      stride      = poolstride,      # stride size
                                      scope       = 'Pool_Layer_%d' % step)

                net = tf.layers.dropout(net, .8)
                                      
                #Increase filters for higher order features        
                filters *= 2
                if debug: print 'After Convolutional Layer ',step,' shape ',net.shape

        with tf.variable_scope('Deep_FC_Layers'):        
            #Flatten the network to 1D
            net = slim.flatten(net, scope='Flatten_Step')
            if debug: print 'After flattening', net.shape

            #Set through a fully connected layer
            for step in xrange(fcnumb):
                net = slim.fully_connected(net, fcsize, scope='FC_Layer_%d' % step)
                if debug: print 'After Fully Connected Layer %d' % step, net.shape
                if trainable:
                    net = slim.dropout(net, keep_prob=0.75, is_training=trainable, scope='fc%d_dropout' % step)
      
        #Set through a final fc layer
        net = slim.fully_connected(net, int(num_class), scope='FC_Final')
        if debug: print 'After Fully Connected Layer Final', net.shape

        end_points = slim.utils.convert_collection_to_dict(outputs_collections)
    
    #Send back the network
    return net,end_points

############
#############

def build2(input_tensor, num_class, trainable, debug):

    outputs_collections = "neunet"

    #Take in input data as 'net' object
    #Expect an input of [BatchSize, Channels, HSize, VSize]
    net = input_tensor
    if debug: print 'Input Tensor: ', input_tensor.shape

    #Define the number of filters to convolve
    filters = config.NETWORK['Filters']
    #Define Kernel size and stride
    kernsize = config.NETWORK['Kernal']
    strider = config.NETWORK['Stride']
    poolstride = config.NETWORK['PoolStride']
    #Define Fully Connected Size
    fcsize = config.NETWORK['FCSize']
    #Define Number of Conv and FC Layers
    clnumb = config.NETWORK['CLNumb']
    fcnumb = config.NETWORK['FCNumb']

    #Loop to set hidden layers in the network
    with tf.variable_scope('Network'):
        with tf.variable_scope('Deep_Conv_Layers'):
            for step in xrange(clnumb):
                #Convolve the network via slim
                net = slim.conv2d(inputs      = net,       # input tensor
                                  num_outputs = filters,   # number of filters/feature maps
                                  kernel_size = kernsize,     # kernel size
                                  stride      = strider,         # stride size
                                  trainable   = trainable, # train or inference
                                  activation_fn = tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer = tf.zeros_initializer(),
                                  scope       = 'Conv_Layer_%d' % step)

                #Extra Stride=1 Convs
                net = slim.conv2d(inputs      = net,       # input tensor
                                  num_outputs = filters,   # number of filters/feature maps
                                  kernel_size = kernsize,     # kernel size
                                  stride      = 1,         # stride size
                                  trainable   = trainable, # train or inference
                                  activation_fn = tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer = tf.zeros_initializer(),
                                  scope       = 'Conv_Layer2_%d' % step)
                
                #Max Pool the network
                net = slim.max_pool2d(inputs      = net,    # input tensor
                                      kernel_size = kernsize,  # kernel size
                                      stride      = poolstride,      # stride size
                                      scope       = 'Pool_Layer_%d' % step)

                #net = tf.layers.dropout(net, .75)
                                      
                #Increase filters for higher order features        
                filters *= 2
                if debug: print 'After Convolutional Layer ',step,' shape ',net.shape

        with tf.variable_scope('Deep_FC_Layers'):        
            #Flatten the network to 1D
            net = slim.flatten(net, scope='Flatten_Step')
            if debug: print 'After flattening', net.shape

            if fcnumb > 0:
                #Set through a fully connected layer
                for step in xrange(fcnumb):
                    net = slim.fully_connected(net, fcsize, scope='FC_Layer_%d' % step)
                    fcsize /= 2
                    if debug: print 'After Fully Connected Layer %d' % step, net.shape
                    if trainable:
                        net = slim.dropout(net, keep_prob=0.75, is_training=trainable, scope='fc%d_dropout' % step)
      
        #Set through a final fc layer
        net = slim.fully_connected(net, int(num_class), scope='FC_Final')
        if debug: print 'After Fully Connected Layer Final', net.shape

        end_points = slim.utils.convert_collection_to_dict(outputs_collections)
    
    #Send back the network
    return net,end_points

##########
#########




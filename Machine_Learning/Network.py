#This module is for building networks separately from the main
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#BUILDING THE NETWORK

def build(input_tensor, num_class, trainable, debug):

    outputs_collections = "neunet"

    #Take in input data as 'net' object
    net = input_tensor
    if debug: print 'Input Tensor: ', input_tensor.shape

    #Define the number of filters to convolve
    filters = 32
    #Define Kernel size and stride
    kernsize = [3,3]
    strider = [1,1]
    #Define Fully Connected Size
    fcsize = 4096
    #Define Number of Conv and FC Layers
    clnumb = 3
    fcnumb = 2

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
                                      stride      = 2,      # stride size
                                      scope       = 'Pool_Layer_%d' % step)
                                      

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

##########
#########




#First Physics Analysis Module, Inference Study a la Kazu
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import os,sys,time
start = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#from Logging import mod_log as log
from Configs import config_main as con
from Loaders import mod_psingle as ps
from Machine_Learning import Network as MLN

#######

#Path to Weights
wpath = '/user/jhenzerling/work/neunet/'
snap = 'Full-Run-Save/neunet-4900'

#Infor for IO
batchsize = 50
cnumb = 5
cfg = ps.get_CFG('test')

#Set up IO
tep = larcv_threadio()
tep.configure(cfg)
tep.start_manager(batchsize)
time.sleep(2)
tep.next(store_entries=True,store_event_ids=True)

#Call the dimensions of the data
tedim = tep.fetch_data('test_image').dim()

########

#Set input
rawinput = tf.placeholder(tf.float32,[None,tedim[1]*tedim[2]*tedim[3]],name='raw')
input2d = tf.reshape(rawinput,[-1,tedim[1],tedim[2],tedim[3]],name='input')

#Build the net
net,endp = MLN.build2(input_tensor = input2d, num_class=cnumb, trainable=False, debug = False)
print('Built Net')
#Define Softmax
softmax = tf.nn.softmax(logits=net)

#sess = tf.InteractiveSession()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    init = tf.global_variables_initializer()
    print('Hit Sess')
    # Load weights
    saver = tf.train.Saver()
    saver.restore(sess, snap)
    print('Loaded Weights')
    

    ########
    #Create a CSV of the output for probability analysis
    fname = 'inf1.csv'
    fpath = wpath + 'Physics/' + fname
    
    #Check if inference already done
    if os.path.exists(fpath) != True:
        print('File does not exist yet')
        fout = open(fname,'w')
        fout.write('entry,run,subrun,event,prediction,probability,label,label_probability\n')

        ctr = 0

        #Call number of events
        events = tep.fetch_n_entries()
        print('Using Events: ', events)
        print('Opened File')
        while ctr < events:
            #Set feed for data
            tedat = tep.fetch_data('test_image').data()
            telab = tep.fetch_data('test_label').data()
            tefeed = {rawinput:tedat}
        
            #Run Softmax
            softmaxb = sess.run(softmax,feed_dict=tefeed)
            if (ctr % events/100) == 0:
                print('%f %% completed in %f seconds' % (ctr/events, start - time.time()))
                prevents = tep.fetch_event_ids()
                prentries = tep.fetch_entries()
        
            #Store in csv
            for i in xrange(len(softmaxb)):
                softmaxarr = softmaxb[i]
                sentry = prentries[i]
                sevent = prevents[i]
            
                pred = np.argmax(softmaxarr)
                pprob = softmaxarr[pred]
                labe = np.argmax(telab[i])
                plab = softmaxarr[labe]
            
                dstring = '%d,%d,%d,%d,%d,%g,%d,%g\n' % (sentry,sevent.run(),sevent.subrun(),
                                                         sevent.event(),pred,pprob,labe,plab)
                fout.write(dstring)
            
                ctr+=1
                if ctr == events:
                    break
            if ctr == events:
                break
                    
            tep.next(store_entries=True,store_event_ids=True)
                    
        fout.close()
        print('Closed File')

    else:
        print('File Already Exists')


    #Create Grid to look at results
    df = pd.read_csv(fpath)
    df.describe()
    
    #Plot the results to see softmax for different PIDS
    particles = ['electron','gamma','muon','pion','proton']
    fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
    for index, particle in enumerate(particles):
        #Pick 1st hist for group by pred. label and 2nd for truth label
        #sub_df = df.query('prediction==%d' % index)
        #hist, _ = np.histogram(sub_df.probability.values, bins=25, range=(0.,1.))
        
        sub_df = df.query('label==%d' % index)
        hist, _ = np.histogram(sub_df.label_probability.values, 
                               bins=25, range=(0.,1.), 
                               weights=[1./sub_df.index.size] * sub_df.index.size)
        
                        
                        
        # Plot
        label = '%s (%d events)' % (particle,sub_df.index.size)
        plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
        # Decoration!
        plt.tick_params(labelsize=20)
        plt.xlabel('Softmax Probability',fontsize=20,fontweight='bold',fontname='Georgia')
        plt.ylabel('Fraction of Events',fontsize=20,fontweight='bold',fontname='Georgia')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.grid()
    plt.show()

    coord.request_stop()
    coord.join(threads)
    end = time.time()
    print('Run Time: %d seconds' % (end - start))

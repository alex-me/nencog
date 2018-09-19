"""
A main to train scnn_cifar.py in background
"""

import  scnn_cifar

scnn_cifar.background   = True
scnn_cifar.setup()
data_set        = scnn_cifar.read_data_sets()
images, labels  = data_set[ 'train' ]
scnn_cifar.train_nn( images, labels )

"""
neural architecture definition
can be imported into cnn_lava.py

alex    Nov 2018

CONSTRAINTS on the nn_arch format:
- items "mask_size", "features", "pool_size", "pool_stride", "norm" are all parameters for
  the convolutional layers, therefore should all be tuples (or lists) of the same size,
  that amount to the number of convolutional layers in the network
- items "full_layer" and "use_dropout" refer to the fully connected layers, therefore should
  be tuples of the same size, that amount to the number of fully connected layers in the network

"""

nn_arch         = {
    "color"         : True,             # color images
    "image_size"    : ( None, None ),   # size of the image
    "mask_size"     : ( 5, 5 ),         # linear size of the convolution mask
    "features"      : ( 6, 4 ),         # number of convolution features for each layer
    "pool_size"     : ( 3, 3 ),         # linear size of maxpool windows
    "pool_stride"   : ( 2, 2 ),         # stride of the maxpool windows
    "norm"          : ( None, None ),   # local response normalization parameters
    "full_layer"    : ( 256, 128 ),     # number of neurons in the fully-connected layers
    "use_dropout"   : ( False, False ), # use dropout in the fully-connected layers
    "classes"       : ( "T", "F" ),     # classe names
    "n_class"       : 2,                # number of classes
}

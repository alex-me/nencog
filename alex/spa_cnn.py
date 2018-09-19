"""
A simple architecture for the classification of a a subset of CIFAR-100,
combining tensorflow convolutional network and nengo SPA representation

alex    Jun 2018

"""

import  os
import  nengo
import  nengo_dl
import  numpy
import  tensorflow              as tf

import  select_cifar
import  cnn_cifar               as cnc
import  cnn

from    tensorflow.contrib.framework    import  get_model_variables
from    tensorflow.contrib.framework    import  get_variables_to_restore
from    tensorflow.contrib.framework    import  assign_from_checkpoint_fn

"""
globals
"""

classes     = None                                  # categories of the CIFAR-100 subset
n_class     = None                                  # number of categories of the CIFAR-100 subset
i_size      = None                                  # number of values in an image
i_shape     = None                                  # image shape
ck_name     = None                                  # filename of the TensorFlow checkpoint

class Vision( object ):
    """
    interface from tensorflow model to nengo_dl.TensorNode
    this class should have the following methods defined
    """
    def post_build( self, sess, rng ):
        print "get_variables_to_restore():"
        print get_variables_to_restore()
        print "get_model_variables():"
        print get_model_variables()
        restore = assign_from_checkpoint_fn( ck_name, get_variables_to_restore()[ 2 : ] )
        restore( sess )

    def __call__( self, t, x ):
        image   = tf.reshape( x, i_shape )
        y       = cnn.set_cnn( image )
        return y


def setup():
    """
    set up main globals
    """
    global classes
    global n_class
    global i_size
    global i_shape
    global ck_name
    cnc.setup( external=True )
    classes = cnc.nn_arch[ 'classes' ]
    n_class = cnc.nn_arch[ 'n_class' ]
    w, h    = cnc.nn_arch[ 'image_size' ]
    ck_name = cnn.get_chck()
    chn     = 3 if cnc.nn_arch[ 'color' ] else 1
    i_size  = w * h * chn
    i_shape = ( 1, w, h, chn )


def setup_nn( seed=1 ):
    """
    build main nengo network
    """
    nn   = nengo.Network( seed=seed )
    with nn:
        v   = nengo_dl.TensorNode( Vision(), size_in=i_size, size_out=n_class, label="cnn" )
        o   = nengo.Probe( v, label="cnn_result" )
    return nn


def recall_nn( image, nn=None ):
    """
    recall the nengo network on the given image
    """
    img = cnn.img_numpy( image )
    if nn is None:
        nn  = setup_nn()
    v   = nn.nodes[ 0 ]
    o   = nn.probes[ 0 ]
    with nn:
        i   = nengo.Node( output=img.flatten() )
        nengo.Connection( i, v, synapse=None, label="img_to_cnn"  )
    with nengo_dl.Simulator( nn, progress_bar=True ) as sim:
        sim.step()
    return sim.data[ o ][ 0 ]


def r_nn( image ):
    """
    recall the nengo network on the given image
    """
    img = cnn.img_numpy( image )
    with nengo.Network() as nn:
        i   = nengo.Node( output=img.flatten() )
        v   = nengo_dl.TensorNode( Vision(), size_in=i_size, size_out=n_class, label="cnn" )
        nengo.Connection( i, v, synapse=None, label="img_to_cnn"  )
        o   = nengo.Probe( v, label="cnn_result" )
    with nengo_dl.Simulator( nn ) as sim:
        sim.step()
    return sim.data[ o ][ 0 ]


"""
esempio d'uso:

setup()
nn      = setup_nn()
recall_nn( "cifar_png/table_10.png", nn ) 
"""

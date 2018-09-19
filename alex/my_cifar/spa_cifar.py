"""
A simple architecture for the classification of a a subset of CIFAR-100,
combining tensorflow convolutional network and nengo SPA representation

alex    Jun 2018

"""

import  os
import  nengo
import  numpy
import  select_cifar
import  nengo_spa               as spa

"""
globals
"""

verbose     = True                                  # produce display

classes     = None                                  # categories of the CIFAR-100 subset
n_class     = None                                  # number of categories of the CIFAR-100 subset
words       = None                                  # SPA vocabulary


def setup():
    """
    set up main globals
    """
    global classes
    global n_class
    global i_size
    global i_shape
    global ck_name
    global words
    cnc.setup()
    classes = cnc.nn_arch[ 'classes' ]
    n_class = cnc.nn_arch[ 'n_class' ]
    w, h    = cnc.nn_arch[ 'image_size' ]
    ck_name = cnc.get_chck()
    chn     = 3 if cnc.nn_arch[ 'color' ] else 1
    i_size  = w * h * chn
    i_shape = ( 1, w, h, chn )
    words   = spa.Vocabulary( dimensions=n_class )
    vect    = numpy.eye( n_class )
    for w, v in zip( classes, vect ):
        words.add( w.upper(), v )


def setup_nn( seed=1 ):
    """
    build main nengo network
    """
    nn   = nengo.Network( seed=seed )
    with nn:
        v   = nengo_dl.TensorNode( Vision(), size_in=i_size, size_out=n_class, label="cnn" )
        vp  = nengo.Probe( v, label="cnn_result" )
    return nn


def recall_nn( image, nn=None, t=0.2 ):
    """
    recall the nengo network on the given image
    """
    if classes is None:
        setup()
    img = cnc.read_image( image )
    if nn is None:
        nn  = setup_nn()
    v   = nn.nodes[ 0 ]
    vp  = nn.probes[ 0 ]
    with nn:
        i   = nengo.Node( output=img.flatten() )
        nengo.Connection( i, v, synapse=None, label="img_to_cnn"  )
        c   = nengo.Config( nengo.Ensemble )
        c[ nengo.Ensemble ].neuron_type = nengo.Direct()
        with c:
# NOTE: for simplicity n_class even is supposed
            o   = spa.State( vocab=words, subdimensions=n_class/2, label="spa_state" )
        nengo.Connection( v, o.input )
    with nengo_dl.Simulator( nn, progress_bar=True ) as sim:
        sim.run( t )
    return sim.data[ vp ]


def pr_class( sim_data ):
    """
    print the classification of the image
    """
    r   = sim_data[ -1 ]
    pr  = 1 + ( r - r.max() ) / r.ptp()
    pr  = pr / pr.sum()
    cnc.print_2class( pr )



"""
esempio d'uso:

"""

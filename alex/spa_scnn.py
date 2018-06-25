"""
A simple architecture for the classification of a a subset of CIFAR-100,
combining tensorflow convolutional network and nengo SPA representation


usage example:
setup()
res = recall_nn( test_i )
pr_class( res )


alex    Jun 2018

"""

import  os
import  nengo
import  nengo_dl
import  numpy
import  tensorflow              as tf
import  tensorflow.contrib.slim as slim

import  select_cifar
import  scnn_cifar              as cnc

import  nengo_spa               as spa

"""
globals
"""

verbose     = True                                  # produce display

classes     = None                                  # categories of the CIFAR-100 subset
n_class     = None                                  # number of categories of the CIFAR-100 subset
i_size      = None                                  # number of values in an image
i_shape     = None                                  # image shape
ck_name     = None                                  # filename of the TensorFlow checkpoint
vocabs      = None                                  # vocabularies
nn          = None                                  # neural network object

test_i      = "cifar_png/woman_01.png"              # test image

class Vision( object ):
    """
    interface from tensorflow model to nengo_dl.TensorNode
    this class should have the following methods defined
    """
    def post_build( self, sess, rng ):
        restore = slim.assign_from_checkpoint_fn( ck_name, slim.get_model_variables() )
        restore( sess )

    def __call__( self, t, x ):
        image   = tf.reshape( x, i_shape )
        y       = cnc.setup_nn( image, dropout_probability=1.0 )
        return y


class Vocab( object ):
    """
    create the vocabularies in use by the model
    """
    def v_words( self ):
        """
        create the main vocabulary with the names of the objects selected among
        CIFAR categories, and extract the subsets of objects to be combined in
        sentences
        """
        self.words  = spa.Vocabulary( dimensions=n_class )
        vect        = numpy.eye( n_class )
        for w, v in zip( classes, vect ):
            self.words.add( w.upper(), v )
        self.word1  = self.words.create_subset( self.obj1 )
        self.word2  = self.words.create_subset( self.obj2 )

    def v_pre( self ):
        """
        create the preposition vocabulary
        """
        self.vpre   = spa.Vocabulary( dimensions=n_class )
        self.vpre.populate( ';'.join( self.pre ) )

    def v_sent( self ):
        """
        create the vocabulary of short sentences, combining names with
        propositions
        """
        self.vsent  = spa.Vocabulary( dimensions=n_class )
        for p in self.pre:
            wp  = spa.translate( self.vpre[ p ], self.words, populate=False )
            for o1 in self.obj1:
                wo1 = self.words[ o1 ] * wp
                for o2 in self.obj2:
                    s   = wo1 + self.words[ o2 ]
                    k   = "{}_{}_{}".format( o1, p, o2 )
                    self.vsent.add( k, s.v )

    def __init__( self ):
        """
        initialize special words like prepositions in use, and the subset of objects for which
        syntactic combinations will be kept in the vocabulary
        """
        self.pre    = ( 'WITH', 'ON' )
        self.obj1   = ( 'TABLE', 'BED', 'CHAIR' )
        self.obj2   = ( 'CUP', 'PLATE', 'BOTTLE' )
        self.v_words()
        self.v_pre()
        self.v_sent()


class Nn( object ):
    """
    build the neural network
    """

    def probe( self, node, label ):
        """
        insert a Nengo probe
        """
        if label in self.probes:
            raise ValueError( "probe's label {} already in use".format( label ) )
        self.probes[ label ]    = nengo.Probe( node, label=label )

    def node( self, node, label, ntype='tf' ):
        """
        insert a Nengo node
        """
        if label in self.nodes:
            raise ValueError( "node's label {} already in use".format( label ) )
        if ntype == 'tf':
            self.nodes[ label ]    = nengo_dl.TensorNode( node, size_in=i_size, size_out=n_class, label=label )
        if ntype == 'nengo':
            self.nodes[ label ]    = nengo.Node( node, size_in=i_size, size_out=n_class, label=label )

    def __init__( self, seed=1 ):
        """
        build the nengo network
        """
        self.probes = {}
        self.nodes  = {}
        self.net    = spa.Network( seed=seed )
        with self.net:
            self.node( Vision(), "cnn1" )
            self.probe( self.nodes[ "cnn1" ], "cnn1_result" )
            c   = nengo.Config( nengo.Ensemble )
            c[ nengo.Ensemble ].neuron_type = nengo.Direct()
            with c:
                o1  = spa.State( vocab=vocabs.words, subdimensions=n_class/2, label="obj1_state" )
            nengo.Connection( self.nodes[ "cnn1" ], o1.input )
            self.probe( o1.output, "obj1_result" )


def setup():
    """
    set up main globals
    """
    global classes
    global n_class
    global i_size
    global i_shape
    global ck_name
    global vocabs
    global nn
    cnc.setup()
    classes = cnc.nn_arch[ 'classes' ]
    n_class = cnc.nn_arch[ 'n_class' ]
    w, h    = cnc.nn_arch[ 'image_size' ]
    ck_name = cnc.get_chck()
    chn     = 3 if cnc.nn_arch[ 'color' ] else 1
    i_size  = w * h * chn
    i_shape = ( 1, w, h, chn )
    vocabs  = Vocab()
    nn      = Nn()


def recall_nn( image, t=0.2 ):
    """
    recall the nengo network on the given image
    """
    if classes is None:
        setup()
    img = cnc.read_image( image )
    v   = nn.nodes[ "cnn1" ]
    with nn.net:
        i   = nengo.Node( output=img.flatten() )
        nengo.Connection( i, v, synapse=None, label="img_to_cnn"  )
    with nengo_dl.Simulator( nn.net, progress_bar=True ) as sim:
        sim.run( t )
    return sim.data


def pr_class( sim_data, img=1 ):
    """
    print the classification of an image presented to the network
    img is the index of the image in the sequence [1,2]
    """
    probe   = nn.probes[ "obj{:1d}_result".format( img ) ]
    idx     = sim_data[ probe ][ -1 ].argsort()
    print( "image classified as   {:20s}".format( vocabs.words.keys()[ idx[ -1 ] ] ) )
    print( "second possibility is {:20s}".format( vocabs.words.keys()[ idx[ -2 ] ] ) )


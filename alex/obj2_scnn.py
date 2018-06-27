"""
A simple architecture for the interpretation of two images taken from
the same scene, in which objects are related with prepositions

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
n_imgs      = 2                                     # number of images as visual stimuli

test_i1     = "cifar_png/plate_03.png"              # test image
test_i2     = "cifar_png/table_01.png"              # test image

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
            size_out            = n_imgs * n_class
            self.nodes[ label ] = nengo_dl.TensorNode( node, size_in=i_size, size_out=size_out, label=label )
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
            self.node( Vision(), "cnn" )
            self.probe( self.nodes[ "cnn" ], "cnn_result" )
            c   = nengo.Config( nengo.Ensemble )
            c[ nengo.Ensemble ].neuron_type = nengo.Direct()
            with c:
                o1  = spa.State( vocab=vocabs.words, subdimensions=n_class/2, label="obj1_state" )
                o2  = spa.State( vocab=vocabs.words, subdimensions=n_class/2, label="obj2_state" )
            s_on    = spa.State( vocab=vocabs.vsent, subdimensions=n_class/2, label="sentence_ON" )
            nengo.Connection( self.nodes[ "cnn" ][ : n_class ], o1.input )
            nengo.Connection( self.nodes[ "cnn" ][ n_class : ], o2.input )
            on  = spa.translate( vocabs.vpre[ 'ON' ], vocabs.words, populate=False )
            spa.translate( o1 * on  + o2, vocabs.vsent, populate=False ) >> s_on
            self.probe( o1.output, "obj1_result" )
            self.probe( o2.output, "obj2_result" )
            self.probe( s_on.output, "s_ON_result" )


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
    i_size  = n_imgs * w * h * chn
    i_shape = ( n_imgs, w, h, chn )
    vocabs  = Vocab()
    nn      = Nn()


def recall_nn( images, t=0.2 ):
    """
    recall the nengo network on the given image
    """
    if classes is None:
        setup()
    imgs    = [ cnc.read_image( i ).flatten() for i in images ]
    imfl    = numpy.concatenate( imgs )
    v       = nn.nodes[ "cnn" ]
    with nn.net:
        i   = nengo.Node( output=imfl )
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



def pr_sent( sim_data ):
    """
    print the most likely sentence associated with two images
    """
    probe   = nn.probes[ "s_ON_result" ]
    simil   = spa.similarity( sim_data[ probe ][ -1 ], vocabs.vsent )
    idx     = simil.argsort()
    print( "most likely sentence  {:20s}".format( vocabs.vsent.keys()[ idx[ -1 ] ] ) )
    print( "second possibility is {:20s}".format( vocabs.vsent.keys()[ idx[ -2 ] ] ) )


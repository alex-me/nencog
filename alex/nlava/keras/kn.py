"""
#############################################################################################################

tests of Keras-Nengo integration

    alex   2019

#############################################################################################################
"""

import  os
import  numpy

import  nengo
import  nengo_dl

import  tensorflow          as tf
import  search_lava         as sl

from keras.models           import clone_model

"""
globals
"""

verbose     = True                                  # produce display

i_size      = None                                  # number of values in an image
i_shape     = None                                  # image shape
ck_name     = None                                  # filename of the TensorFlow checkpoint
vocabs      = None                                  # vocabularies
nn          = None                                  # neural network object
categories  = [                                     # categories of possible objects
    'person', 'chair', 'bag', 'telescope' ]
persons     = [ 'Andrei', 'Danny', 'Yevgeni' ]      # name of possible persons
preposition = [ 'with' ]                            # prepositions
test_str    = \
    """00022-10300-10390    Danny approached the chair with a yellow bag"""
test_img    = "00022-10300-10390"
shapes      = {}                                    # dictionary with the shapes of an array for each category
coords      = {}                                    # dictionary with subwindows coordinates for each category



class Kinter( object ):
    """
    interface to the Keras model for a category
    """
    def pre_build( self, *args ):
        """
        this instruction is necessary, because keras.models.clone_model instantiate the
        graph of the neural model again, and this way it become included in nengo_dl
        """
        self.model  = clone_model( self.model )

    def post_build( self, *args ):
        """
        the cloning of the model, done by pre_build() has also the effect of loosing all weights,
        that should be loaded again from the original model (and saved in self.weights by __init__()
        """
        self.model.set_weights( self.weights )

    def __call__( self, t, x ):
        """
        """
        winds   = tf.reshape( x, self.shape )
        y       = self.model.call( winds )
        y       = y[ :, 0 ]
        y       = tf.reshape( y, ( 1, y.shape[ 0 ] ) )
        return y

    def __init__( self, category="person" ):
        """
        initalize the parameters of the interface depending on the category of objects
        """
        self.category   = category
        self.shape      = shapes[ category ]
        self.model      = models[ category ]
        self.weights    = self.model.get_weights()


class Nn( object ):
    """
    build the neural network in Nengo
    an instance of the class has as main attributes the following:
        slef.net        the Nengo network
        self.nodes      a dictionary of the Nengo nodes defined in the network
        self.probes     a dictionary of the Nengo probe objects defined in the network
    """

    def probe( self, node, label ):
        """
        insert a Nengo probe
        """
        if label in self.probes:
            raise ValueError( "probe's label {} already in use".format( label ) )
        self.probes[ label ]    = nengo.Probe( node, label=label )

    def inode( self, category='person' ):
        """
        insert a Nengo placeholder input node
        """
        label   = "input_" + category
        if label in self.nodes:
            raise ValueError( "node's label {} already in use".format( label ) )
        size_in             = numpy.prod( shapes[ category ] )
        output              = numpy.ones( ( size_in, ) )
        self.nodes[ label ] = nengo.Node( output=output, label=label )

    def knode( self, category='person' ):
        """
        insert a Nengo node for Keras interface
        """
        if category in self.nodes:
            raise ValueError( "node's label {} already in use".format( category ) )
        size_in             = numpy.prod( shapes[ category ] )
        size_out            = shapes[ category ][ 0 ]
        label               = category
        node                = Kinter( category )
        self.nodes[ label ] = nengo_dl.TensorNode( node, size_in=size_in, size_out=size_out, label=label )

    def nnode( self, label, size_in, size_out ):
        """
        insert an ordinary Nengo node
        """
        if label in self.nodes:
            raise ValueError( "node's label {} already in use".format( label ) )
        self.nodes[ label ]    = nengo.Node( node, size_in=size_in, size_out=size_out, label=label )

    def __init__( self, seed=1, label="KerNengo", cats=None ):
        """
        build the nengo network
        there will be as many input and Keras node objects as the categories to be searched
        in the image. The categories can be given in a list as cats argoments, otherwise the
        global categories is used
        """
        self.probes = {}
        self.nodes  = {}
        self.net    = nengo.Network( seed=seed, label=label )
        if cats is None:
            self.categories = categories
        else:
            self.categories = cats
        with self.net:
            for c in self.categories:
                inp     = "input_" + c
                res     = c + "_result"
                self.inode( c )
                self.knode( c )
                self.probe( self.nodes[ c ], res )
                nengo.Connection( self.nodes[ inp ], self.nodes[ c ], synapse=None )


def read_sentence( sentence ):
    """
    read one sentence and extract the image file name, the objects mentioned in the sentence
    and the object which is complment of the preposition in the sentence

    return the file name, the category of the complment, and a dictionary with numer of instances
    for all the categories mentioned in the sentence
    """

    cats    = {}
    words   = sentence.split()
    fimg    = words.pop( 0 )
    pre     = False
    comp    = None
    for w in words:
        if w in preposition:
            pre = True
        if w in persons:
            word    = "person"
        else:
            word    = w
        if word in categories:
            if word in cats.keys():
                cats[ word ]   += 1
            else:
                cats[ word ]   = 1
            if pre:
                comp            = word

    return fimg, comp, cats


def read_image( fname ):
    """
    read one LAVA image and generate batches of subwindows, specific for each category,
    in addition, validate the globals shapes and coords to be used by nengo TensorNode
    and further processes

    return a dictionary with batches and coordinates for all categories
    """
    global shapes
    global coords

    inputs  = {}
    img     = sl.read_image( fname )
    for c in categories:
        inputs[ c ] = sl.scan_image( img, c )
        shapes[ c ] = inputs[ c ][ "windows" ].shape
        coords[ c ] = inputs[ c ][ "coordinates" ]
        
    return inputs


def setup():
    """
    set up main globals

    NOTE that it is necessary to call read_image() on a test image in order to
    validate the shapes of the tensor with subwindows for the various categories,
    and the definition of Nn() depends on such shapes
    """
    global models
    global nn

    sl.setup()
    models  = sl.models
    _       = read_image( test_img )
    nn      = Nn()


def in_windows( imgs ):
    """
    distribute the windows read from a LAVA image as input to the neural model
    """
    data    = {}
    for c in nn.categories:
        img             = imgs[ c ][ "windows" ]
        shape           = img.shape
        img             = img.reshape( ( -1, numpy.prod( shape ) ) )
        img             = img[ :, None, : ]
        i_in            = nn.nodes[ "input_" + c ]
        data[ i_in ]    = img
    return data


def hits( sim_data, category="person", nhits=10 ):
    """
    find the coordinates of the nhits best hits for the given category in the response
    of a simulation
    """
    probe   = nn.probes[ category + "_result" ]
    res     = sim_data[ probe ][ 0 ]
    idx     = res.argsort()
    coo     = numpy.array( coords[ category ] )
    return coo[ idx[ -nhits : ] ]


def find_hits( sim_data, nhits=10 ):
    """
    find the coordinates of the nhits best hits for all categories in the response
    of a simulation
    """
    h   = {}
    for c in nn.categories:
        h[ c ]  = hits( sim_data, category=c, nhits=nhits )
    return h


def recall_nn( image ):
    """
    recall the nengo network on the given image, and return the simulation data
    """
    if models is None:
        return None
    imgs    = read_image( image )
    data    = in_windows( imgs )
    with nengo_dl.Simulator( nn.net, progress_bar=True ) as sim:
        sim.step( data=data )
    return sim.data


def recall_sentence( sentence ):
    """
    given a LAVA sentence (including the image name), extract from the sentence the
    objects of interest, apply the nengo network on the relevant image to search the
    objects, and return a dictionary with the coordinates where the objects are found
    """
    if models is None:
        return None
    img, comp, cats = read_sentence( sentence )
    if comp is None:
        return None
    nn.categories   = list( cats.keys() )
    sim             = recall_nn( img )
    return find_hits( sim )


def disambiguate( sentence ):
    """
    given a LAVA sentence (including the image name), extract from the sentence the
    objects of interest, apply the nengo network on the relevant image to search the
    objects, and from the coordinates where the objects are found find the best syntactic
    matching for the preposition "where"
    """
    if models is None:
        return None
    img, comp, cats = read_sentence( sentence )
    if comp is None:
        return None
    nn.categories   = list( cats.keys() )
    sim             = recall_nn( img )
    hits            = find_hits( sim, nhits=1 )
    c0              = hits[ comp ][ 0 ]
    dist            = 100000.0
    link            = None
    for cat in hits.keys():
        if cat == comp: continue
        c1  = hits[ cat ][ 0 ]
        d   = numpy.linalg.norm( c1 - c0 )
        if d < dist:
            link    = cat
            dist    = d

    return link + " <- with -> " + comp

"""
usage example:
setup()
disambiguate( test_str )
"""

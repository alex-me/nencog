"""
#############################################################################################################

context model

disambiguate sentences of the LAVA dataset analyzing images, integrating Keras witgh Nengo SPA

model definition, model recall, post-processing of model output for disambiguation

    alex   2019

#############################################################################################################
"""

import  os
import  numpy
import  h5py

import  nengo
import  nengo_dl
import  nengo_spa           as spa
import  tensorflow          as tf

import  lava_geo            as lg

from    keras.models            import load_model, clone_model
from    matplotlib              import pyplot
from    mpl_toolkits.mplot3d    import Axes3D

"""
globals
"""

NO_GPU      = True                                  # force disabling GPU usage

verbose     = 2                                     # produce Nengo progress bar
seed        = 1                                     # seed for Nengo networks
batch_pro   = False                                 # probabilities are batched, and should be read externally
mod_dir     = "../keras/model_dir"                  # directory with Keras models
pro_dir     = "prob"                                # directory where to read probabilities in .h5 files
mname       = "nn_best.h5"                          # standard name of model file
models      = {}                                    # dictionary with Keras models
timestep    = 0.002                                 # duration in seconds of a Nengo timestep
n_tsteps    = 50                                    # number of timesteps in the Nengo simulation
methods     = ( "CLOSENESS", "SIMILARITY" )         # methods actually implemented in kspa_model for the
                                                    # evaluation of spatial closeness fo objects
method      = "CLOSENESS"                           # default method in use

vocabs      = None                                  # vocabularies
nn          = None                                  # neural network object
persons     = [ 'Andrei', 'Danny', 'Yevgeni' ]      # name of possible persons
preposition = [ 'with' ]                            # prepositions
colors      = {                                     # colors for plots
    'person'    : '#3cb371',
    'chair'     : '#ff69b4',
    'bag'       : '#dc143c',
    'telescope' : '#8a2be2'
}
plt_alpha   = 0.6                                   # transparency of surfaces for plot
test_str    = \
    """00022-10300-10390       Danny approached the chair with a yellow bag"""  # chair <-> bag
test_img    = "00022-10300-10390"
test_str1   = \
    """00022-10090-10190       Danny approached the chair with a yellow bag"""  # persons <-> bag
test_img1   = "00022-10090-10190"



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
        this is the main function, that actually calls the Keras model
        the input is a batch of all subwindows for one category, but
        the reusult is averaged over the vertical dimension, so that the
        output vector is a proabability of finding an object of a category
        along the horizontal dimension
        NOTE: even if the function does not take time into consideration, the
        shape of the output tensor should be augmented for the time dimension
        """
        winds   = tf.reshape( x, self.shape )
        y       = self.model.call( winds )
        y       = y[ :, 0 ]
        y       = tf.reshape( y, lg.n_coords[ self.category ] )
#       y       = tf.reduce_mean( y, axis=1 )
        y       = tf.reduce_max( y, axis=1 )
        y       = tf.reshape( y, ( 1, y.shape[ 0 ] ) )
        return y

    def __init__( self, category="person" ):
        """
        initalize the parameters of the interface depending on the category of objects
        """
        self.category   = category
        self.shape      = lg.shapes[ category ]
        self.model      = models[ category ]
        self.weights    = self.model.get_weights()


class Nn( object ):
    """
    build the neural network in Nengo, composed by two models: one using Nengo-DL as
    interface with Keras, and one using Nengo-SPA for post-processing the vision data

    an instance of the class has as main attributes the following:
        slef.dl         the Nengo-DL network
        slef.spa        the Nengo-SPA network
        self.nodes      a dictionary of the Nengo nodes defined in the network
        self.probes     a dictionary of the Nengo probe objects defined in the network
        self.states     a dictionary of the Nengo spa.State objects
        self.prob       a dictionary with categories as keys and current probabilities as values
    """

    def probe( self, node, label, net='dl' ):
        """
        insert a Nengo probe
        """
        label   = net + '_' + label
        if label in self.probes:
            raise ValueError( "probe's label {} already in use".format( label ) )
        self.probes[ label ]    = nengo.Probe( node, label=label )

    def inode( self, category='person', net='dl' ):
        """
        insert a Nengo placeholder input node, taking into account the correct size
        and type depending on the network
        """
        label   = "{}_input_{}".format( net, category )
        if label in self.nodes:
            raise ValueError( "node's label {} already in use".format( label ) )
        if net=='dl':
            size_in             = numpy.prod( lg.shapes[ category ] )
            output              = numpy.ones( ( size_in, ) )
        elif net=='spa':
            output              = lambda t: self.prob[ category ]
        else:
            raise ValueError( "unkown netowrk type {} in inode()".format( net ) )
        self.nodes[ label ] = nengo.Node( output=output, label=label )

    def knode( self, category='person' ):
        """
        insert a Nengo node for Keras interface
        """
        label               = 'dl_' + category
        if label in self.nodes:
            raise ValueError( "node's label {} already in use".format( label ) )
        size_in             = numpy.prod( lg.shapes[ category ] )
        size_out            = lg.n_coords[ category ][ 0 ]
        node                = Kinter( category )
        self.nodes[ label ] = nengo_dl.TensorNode( node, size_in=size_in, size_out=size_out, label=label )

    def nnode( self, label, size_in, size_out, net='dl' ):
        """
        insert an ordinary Nengo node
        """
        label   = net + '_' + label
        if label in self.nodes:
            raise ValueError( "node's label {} already in use".format( label ) )
        self.nodes[ label ]    = nengo.Node( node, size_in=size_in, size_out=size_out, label=label )

    def get_node( self, label, net='dl' ):
        """
        return a Nengo node
        """
        label   = net + '_' + label
        if label not in self.nodes:
            raise ValueError( "node {} not found".format( label ) )
        return self.nodes[ label ]

    def get_probe( self, label, net='dl' ):
        """
        return a Nengo node
        """
        label   = net + '_' + label
        if label not in self.probes:
            raise ValueError( "probes {} not found".format( label ) )
        return self.probes[ label ]

    def state( self, label ):
        """
        insert a SPA state
        NOTE: the use of spa.State instead of vocabulary keys is that it looks impossible to
        connect non-spa objects to vocabulary items
        conversely, it seems impossible to add spa.State objects into the vocabulary
        """
        if label in self.states:
            raise ValueError( "node's label {} already in use".format( label ) )
        s                       = spa.State( vocab=self.voc, subdimensions=1, label=label )
        self.states[ label ]    = s

    def __init__( self, seed=1, label="KerNengo", cats=None ):
        """
        build the nengo network
        there will be as many input and Keras node objects as the categories to be searched
        in the image. The categories can be given in a list as cats argoments, otherwise the
        global categories is used
        """
        self.probes = {}
        self.nodes  = {}
        self.states = {}
        self.prob   = {}
        self.v_dim  = list(lg.n_coords.values())[ 0 ][ 0 ]
        self.voc    = spa.Vocabulary( self.v_dim )
        self.dl     = nengo.Network( seed=seed, label='DL'+label )
        self.spa    = spa.Network( seed=seed+1, label='SPA'+label )

        if cats is None:
            self.categories = lg.categories
        else:
            self.categories = cats

        # initialize probabilities
        for c in self.categories:
            self.prob[ c ]  = numpy.zeros( ( self.v_dim, ) )

        """
        this is the part intended for using a vocabolary, but I have found no ways to use it
        (see NOTE in state())
        n_cat       = len( self.categories )
        parse_str   = (n_cat - 1 ) * "WHERE{}; " + "WHERE{}" 
        spointers   = parse_str.format( *self.categories )
        self.voc.populate( spointers )
        """

        with self.dl:
            net = 'dl'
            for c in self.categories:
                inp     = "input_" + c
                self.inode( c, net=net )
                self.knode( c )
                nengo.Connection( self.get_node( inp, net=net ), self.get_node( c, net=net ), synapse=None )
                self.probe( self.get_node( c, net=net ), c, net=net )

        with self.spa:
            net = 'spa'
            cfg = nengo.Config( nengo.Ensemble )
            cfg[ nengo.Ensemble ].neuron_type   = nengo.Direct()
            with cfg:
                for c in self.categories:
                    inp     = "input_" + c
                    self.inode( c, net=net )
                    where   = c + "_where"
                    self.state( c )
                    nengo.Connection( self.get_node( inp, net=net ), self.states[ c ].input )
                    self.probe( self.states[ c ].output, where, net=net )



def setup():
    """
    set up the external information in lava_geo, read the Keras models,
    setup the Nengo-nengo_dl-nengo_spa model
    """
    global models
    global nn

    if NO_GPU:
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]  = ""

    lg.setup()

    for c in lg.categories:
        models[ c ] = load_model( os.path.join( mod_dir, c, mname ) )

    nn      = Nn( seed=seed )


def read_sentence( sentence ):
    """
    read one sentence and extract the image file name, the objects mentioned in the sentence
    and the object which is complment of the preposition in the sentence

    return the file name, the category of the complement, and a dictionary with numer of instances
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
        if word in lg.categories:
            if word in cats.keys():
                cats[ word ]   += 1
            else:
                cats[ word ]   = 1
            if pre:
                comp            = word

    return fimg, comp, cats


def ambigue( comp, cats ):
    """
    given the complment and the list of all categories in a sentence,
    return the two alternative possible object to bind to the ocmplement

    NOTE: this function is useful only when the three relevent objects in
    the scene are different
    """
    a   = []
    for c in cats.keys():
        if c == comp: continue
        a.append( c )

    return a


def in_windows( imgs ):
    """
    distribute the windows read from a LAVA image as input to the neural model
    """
    data    = {}
    for c in nn.categories:
        img             = imgs[ c ]
        length          = numpy.prod( img.shape )
        img             = img.reshape( ( length, ) )
        img             = img.reshape( ( 1, 1, length ) )
        i_in            = nn.get_node( "input_" + c, net='dl' )
        data[ i_in ]    = img
    return data


def closest( comp, amb0, amb1 ):
    """
    find which of amb0, amb1 is closest to comp
    """
    l   = [ ( comp.argmax() - amb0.argmax() )**2, ( comp.argmax() - amb1.argmax() )**2 ]
    l   = numpy.array( l )
    return l.argmin()


def find_ambigue( sim_data, comp, amb ):
    """
    find the best match for WITH, given the complement comp and the two ambigue
    categories amb
    """
    s_comp  = sim_data[ nn.get_probe( comp + '_where', net='spa' ) ][ -1 ]
    s_amb0  = sim_data[ nn.get_probe( amb[ 0 ] + '_where', net='spa' ) ][ -1 ]
    s_amb1  = sim_data[ nn.get_probe( amb[ 1 ] + '_where', net='spa' ) ][ -1 ]
    if method == 'SIMILARITY':
        r   = spa.similarity( s_comp, [ s_amb0, s_amb1 ], normalize=True )
        return  r.argmax()
    if method == 'CLOSENESS':
        return  closest( s_comp, s_amb0, s_amb1 )


def recall_dl( data ):
    """
    recall the nengo DL network and return the simulation data
    """
    p   = {}
    pb  = verbose >= 2
    with nengo_dl.Simulator( nn.dl, progress_bar=pb ) as sim:
        sim.step( data=data )
        for c in nn.categories:
            probe   = nn.get_probe( c, 'dl' )
            p[ c ]  = sim.data[ probe ][ 0 ]
    return p


def read_prob( image ):
    """
    read probability data precomputed and stored in h5 files
    """
    fname   = os.path.join( pro_dir, image + '.h5' )
    if not os.path.isfile( fname ):
        print( "Error: file {} not found".format( fname ) )
        sys.exit()
    f       = h5py.File( fname, 'r' )
    p       = {}
    for c in nn.categories:
        if c not in f.keys():
            print( "Error: category {} not found in h5 file".format( c ) )
            sys.exit()
        # NOTE: this fancy coding is required by last h5py versions, the
        # older f[ c ].value is now deprecated
        p[ c ]  = f[ c ][ () ]

    return p


def recall_spa( prob ):
    """
    recall the nengo SPA network and return the simulation data

    input:  [dir] arrays of probabilities for each category
    """
    pb  = verbose >= 2
    for c in nn.categories:
        nn.prob[ c ]    = prob[ c ]
    with nengo.Simulator( nn.spa, dt=timestep, progress_bar=pb ) as sim:
        sim.run( n_tsteps * timestep )
        return sim


def recall_nn( image ):
    """
    recall the nengo network on the given image, and return the simulation data
    """
    if models is None:
        return None
    if batch_pro:
        prob    = read_prob( image )
    else:
        imgs    = lg.read_scans( image )
        data    = in_windows( imgs )
        prob    = recall_dl( data )
    sim     = recall_spa( prob )
    return sim


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
    amb             = ambigue( comp, cats )
    if comp is None:
        return None
    nn.categories   = list( cats.keys() )
    sim             = recall_nn( img )
    hit             = find_ambigue( sim.data, comp, amb )
    sim.close()
    return amb[ hit ], comp


def plot_spa( sim_data, comp, amb, sentence ):
    """
    find the best match for WITH, given the complement comp and the two ambigue
    categories amb
    """
    fig     = pyplot.figure()
    plt     = fig.add_subplot( 111, projection='3d' )
    time    = numpy.linspace( 0, 1000 * n_tsteps * timestep, n_tsteps )
    space   = lg.x_coordinates
    fname   = "plspa_{}.pdf".format( sentence.split()[ 0 ] )
    x, y    = numpy.meshgrid( time, space )
    xl      = 0.10
    yl      = 0.90
    dyl     = 0.05
    for i, c in enumerate( ( comp, amb[ 0 ], amb[ 1 ] ) ):
        z   = sim_data[ nn.get_probe( c + '_where', net='spa' ) ]
        plt.plot_surface( x, y, z.transpose(), color=colors[ c ], alpha=plt_alpha )
        # NOTE: plot_surface has no label argument, therefore labels are put manually:
        plt.text2D( xl, yl - i * dyl, c, fontsize='x-small', color=colors[ c ], transform=plt.transAxes )
    plt.set_xlabel( 'time [msec]' )
    plt.set_ylabel( 'horiz. space [pixel]' )
    plt.set_zlabel( 'probability' )
    plt.tick_params( labelsize='xx-small' )
    pyplot.savefig( fname )
    pyplot.close()


def evolve_spa( sentence ):
    """
    given a LAVA sentence (including the image name), extract from the sentence the
    objects of interest, apply the nengo network on the relevant image to search the
    objects, and extract the evolution in time of the SPA states
    """
    if models is None:
        return None
    img, comp, cats = read_sentence( sentence )
    amb             = ambigue( comp, cats )
    if comp is None:
        return None
    nn.categories   = list( cats.keys() )
    sim             = recall_nn( img )
    plot_spa( sim.data, comp, amb, sentence )
    sim.close()


def init_stat( comp=( 'bag', 'telescope' ), amb=( 'bag', 'person', 'chair' ) ):
    """
    initialize the data structures used by evaluate()
    comp should include all the possible categories that play the role of complements
    of the prepositions
    the structure class_res is a dictionary with 3 levels:
        - the level of all the possible complements
        - the level of all the possible true categories associate to the complements
        - the level of all the possible predicted categories associate to the complements
    the values at the inner level will be the number of counts, initially are 0
    """
    class_res   = {}

    for c in comp:
        class_res[ c ]  = {}
        for t in amb:
            if t == c: continue
            class_res[ c ][ t ] = {}
            for p in amb:
                if p == c: continue
                class_res[ c ][ t ][ p ] = 0
    
    return class_res


def evaluate( sentences, truths, full=None ):
    """
    given a list of LAVA sentences and a list of corresponding ground truths, evaluate
    the accuracy of the model
    if full is not None, it is assumed as the file descriptor where writing full report
    of the evaluation
    """
    res     = init_stat()

    for s, t in zip( sentences, truths ):
        p, comp                 = disambiguate( s )
        res[ comp ][ t ][ p ]   += 1
        if full is not None:
            s   = s.strip().replace( '\t', ' ' )
            full.write( "{:80s}{:10s}{:10s}\t{}\n".format( s, p, t, t==p ) )
        if verbose:
            print( "done " + s )

    return res

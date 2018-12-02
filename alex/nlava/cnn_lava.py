"""
A simple convolutional architecture for a classifer on a subset of CIFAR-100
The personalized classes, and the directory with the personalized datasets are
imported from select_cifar, the script that generates the CIFAR-100 selection

this program does not use slim, but cnn, a simple wrapper on tensorflow primitives

alex    May 2018

"""


import  pickle
import  gzip
import  os
import  numpy
import  sys
import  tensorflow              as tf

import  cnn
import  pack_lava


"""
main globals
"""
verbose         = 1                     # verbose level
dsdir           = "./data"              # directory where to read datasets
d_prefix        = "lava_"               # common prefix to dataset filenames
m_prefix        = "cnn_"                # common prefix to models
model_dir       = "./model_dir"
learning_rate   = 0.001
epochs          = 50
test_frac       = 0.05                  # fraction of samples to be used as test set
graphs          = {}                    # dictionary of TF graphs in use
sessions        = {}                    # dictionary of TF sessions in use
models          = {}                    # dictionary of TF models in use


def import_arch():
    """
    import an external network definition
    """
    try:
        import neural_arch
        cnn.nn_arch     = neural_arch.nn_arch
        return True
    except ImportError:
        return False


def split_test( data, n ):
    """
    split data into train and test sets
    """
    n_2     = n // 2
    test_y  = data[ : n_2 ]
    test_n  = data[ - n_2 : ]
    test    = numpy.concatenate( ( test_y, test_n ) )
    train   = data[ n_2 : - n_2 ]
    return { "test" : test, "train" : train }


def read_data_set( category='person' ):
    """
    read the dataset of a object category, split into teain and test, and
    reorganize into the format used by cnn
    """
    f       = d_prefix + category + '.gz'
    f       = os.path.join( dsdir, f )
    gf      = gzip.open( f )
    dset    = pickle.load( gf )
    gf.close()

    imgs    = dset[ "images" ]
    labs    = dset[ "labels" ]
    n       = imgs.shape[ 0 ]
    n_test  = pack_lava.divisor * round( 0.05 * ( n / pack_lava.divisor ) )
    i_split = split_test( imgs, n_test )
    l_split = split_test( labs, n_test )
    train   = i_split[ "train" ], l_split[ "train" ]
    test    = i_split[ "test" ], l_split[ "test" ]

    return { "train" : train, "test" : test }


def set_cat( category ):
    """
    set a category as the current one
    """
    if not category in pack_lava.categories:
        if verbose:
            print( "Error: category " + category + " is missing" )
        sys.exit()
    cnn.nn_arch[ "image_size" ] = pack_lava.sizes[ category ]
    cnn.model_name              = m_prefix + category


def setup( category=None ):
    """
    set up main variables
    optionally adapt to a specific category
    """
    if cnn.nn_arch is None:
        if not import_arch():
            if verbose:
                print( "Error: no external neural architecture definition found" )
            sys.exit()
    cnn.weights_dir = model_dir
    if not os.path.isdir( model_dir ):
        os.mkdir( model_dir )
        if verbose:
            print( "model directory " + model_dir + " missing, created" )
    if category is not None:
        set_cat( category )


def train_nn( data_set=None, category='person', save=True ):
    if data_set is None:
        data_set    = read_data_set( category=category )
    session     = cnn.train_cnn( data_set, epochs=epochs, learning_rate=learning_rate, save=save )
    return session


def restore_nn( category='person' ):
    """
    restore the model trained for specific category
    save its graph and session too
    """
    set_cat( category )
    graph                   = tf.Graph()
    graphs[ category ]      = graph
    session                 = cnn.start_session( graph=graph )
    sessions[ category ]    = session
    with graph.as_default():
        nn  = cnn.restore_weights( session )
    models[ category ]      = nn
    return session, nn


def restore_all_nn():
    for category in pack_lava.categories:
        set_cat( category )
        restore_nn( category=category )


def test_nn( data_set=None, category='person' ):
    if data_set is None:
        data_set    = read_data_set( category=category )
    session, nn = restore_nn()
    t           = cnn.test_cnn( session, data_set, nn )
    session.close()
    return t


def recall_batch( images, session=None, nn=None ):
    """
    recall the network on a batch of images, returning probabilities of the object
    to be present in the images
    """
    if not isinstance( images, numpy.ndarray ):
        if verbose:
            print( "Error in recall_batch: invalid format for images" )
        return None
    if len( images.shape ) != 4:    
        if verbose:
            print( "Error in recall_batch: invalid format for images" )
        return None

    if nn is None:
        session, nn = restore_nn()
    r   = cnn.recall( session, images, nn )
    """
    print( "r :" )
    print( r )
    """
    nr  = 1 + ( r - r.max() ) / r.ptp()
    return nr[ :, 0 ]


def recall_cat_batch( images, category='person' ):
    """
    recall the network for a single image on a specified category
    """
    if verbose > 1:
        print( "doing category " + category )
    set_cat( category )
    with graphs[ category ].as_default():
        r   = recall_batch( images, session=sessions[ category ], nn= models[ category ] )
    return r


def recall( image, session=None, nn=None, label=None ):
    """
    recall the network for a single image
    """
    if isinstance( image, numpy.ndarray ):
        img     = image
    else:
        if not isinstance( image, str ):
            if verbose:
                print( "Error in recall: invalid image " + str( image ) )
            return None
        img         = pack_lava.read_image( image, graylevel=False )

    classes = cnn.nn_arch[ "classes" ]
    if nn is None:
        session, nn = restore_nn()
    r   = cnn.recall( session, img, nn )
    nr  = 1 + ( r - r.max() ) / r.ptp()
    nr  = nr / nr.sum()
    inx = nr.argsort()
    print( "image classified as   " + classes[ inx[ -1 ] ] + " with probability " + str( nr[ inx[ -1 ] ] ) )
    print( "second possibility is " + classes[ inx[ -2 ] ] + " with probability " + str( nr[ inx[ -2 ] ] ) )
    if label is not None:
        print( "ground truth is " + classes[ int( label ) ] )


def recall_cat( image, category='person', label=None ):
    """
    recall the network for a single image on a specified category
    """
    set_cat( category )
    with graphs[ category ].as_default():
        recall( image, session=sessions[ category ], nn= models[ category ] )


"""
# to train:
category    = 'person'
setup( category=category )
train_nn( category=category )

# to test using the last checkpoint:
category    = 'person'
setup( category=category )
test_nn( category=category )

# to recall on an image using the last checkpoint:
category    = 'person'
setup( category=category )
recall( "samples/person/f_00203.png" )

"""

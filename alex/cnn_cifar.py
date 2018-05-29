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

import  select_cifar
import  cnn


"""
main globals
"""
verbose         = 1                     # verbose level

nn_arch         = {                     # overall network definition
    "color"         : True,             # color images
    "image_size"    : ( 32, 32 ),       # size of the image
    "mask_size"     : ( 5, 5 ),         # linear size of the convolution mask
    "features"      : ( 4, 4 ),         # number of convolution features for each layer
    "pool_size"     : ( 3, 3 ),         # linear size of maxpool windows
    "pool_stride"   : ( 2, 2 ),         # stride of the maxpool windows
    "norm"          : ( None, {         # local response normalization parameters
        "depth_radius"  : 5,
        "bias"          : 1.0,
        "alpha"         : 0.001 / 9.0,
        "beta"          : 0.75
    } ),                              
    "full_layer"    : ( 256, 128 ),     # number of neurons in the fully-connected layers
    "use_dropout"   : ( False, False ), # use dropout in the fully-connected layers
    "classes"       : None,             # classe names
    "n_class"       : None,             # number of classes
}

model_dir       = "./model_dir"
model_name      = "mycifar"
learning_rate   = 0.001
epochs          = 20


def setup_cifar():
    """
    adapt to the current selection of CIFAR-100
    """
    nn_arch[ "classes" ]    = select_cifar.classes
    nn_arch[ "n_class" ]    = len( select_cifar.classes )


def import_arch():
    """
    import an external network definition
    """
    try:
        import neural_arch
        nn_arch     = neural_arch.nn_arch
        return True
    except ImportError:
        return False


def dense_to_one_hot( l ):
    """
    Convert class labels from scalars to one-hot vectors
    """
    l       = numpy.array( l )
    nl      = l.shape[0]
    i_off   = numpy.arange( nl ) * nn_arch[ 'n_class' ]
    labels  = numpy.zeros(( nl, nn_arch[ 'n_class' ] ) )
    labels.flat[ i_off + l.ravel() ] = 1
    return labels


def norm_images( i ):
    """
    normalize image values om ramge 0..1
    """
    i   = i.astype( numpy.float32 )
    return numpy.multiply( i, 1.0 / 255.0 )


def adj_dataset( d ):
    """
    adjust a CIFAR image dataset, composed of linear arrays of pixes, into standard image tensors
    """
    w, h    = nn_arch[ 'image_size' ]
    chn     = 3 if nn_arch[ 'color' ] else 1
    n       = d.shape[ 0 ]
    dset    = d.reshape( ( n, chn, w, h ) )
    dset    = dset.transpose( ( 0, 2, 3, 1 ) )
    return norm_images( dset )


def read_data_sets():
    """
    read the personalized training and test sets, adjusting the format of the data as neeeded
    """
    data_set            = {}
    path                = select_cifar.cifar_sel

    fname               = select_cifar.cifar_trs + ".gz" 
    f                   = os.path.join( path, fname )
    fd                  = gzip.open( f, 'rb' )
    d                   = select_cifar.my_pickle( fd )
    fd.close()
    train_data          = adj_dataset( d[ 'data' ] )
    train_labels        = dense_to_one_hot( d[ 'labels' ] )

    fname               = select_cifar.cifar_tss + ".gz" 
    f                   = os.path.join( path, fname )
    fd                  = gzip.open( f, 'rb' )
    d                   = select_cifar.my_pickle( fd )
    fd.close()
    test_data           = adj_dataset( d[ 'data' ] )
    test_labels         = dense_to_one_hot( d[ 'labels' ] )

    data_set[ 'train' ] = ( train_data, train_labels )
    data_set[ 'test' ]  = ( test_data, test_labels )

    return data_set


def setup( external=True ):
    """
    set up main variables
    NOTE that the sequence of expressions is crucial, in order to preserve a valid nn_arch
    and export it correctly to cnn
    """
    if external:
        if not import_arch():
            if verbose:
                print "warning: no external neural architecture definition found, using internal"
    setup_cifar()
    cnn.nn_arch     = nn_arch
    cnn.weights_dir = model_dir
    cnn.model_name  = model_name
    if not os.path.isdir( model_dir ):
        os.mkdir( model_dir )
        if verbose:
            print( "model directory " + model_dir + " missing, created" )


def train_nn( data_set=None, save=True ):
    if data_set is None:
        data_set    = read_data_sets()
    nn          = cnn.train_cnn( data_set, epochs=epochs, learning_rate=learning_rate, save=save )
    return nn


def restore_nn( session=None ):
    if session is None:
        session = cnn.start_session()
    return session, cnn.restore_weights( session )


def test_nn( data_set=None):
    if data_set is None:
        data_set    = read_data_sets()
    session, nn = restore_nn()
    t           = cnn.test_cnn( session, data_set, nn )
    session.close()
    return t


def recall( image, session=None, nn=None, label=None ):
    """
    recall the network for a single image, if there is a database the image can be given
    as record number, otherwise should be a path name for an image file
    """
    if not isinstance( image, str ):
        print( "Error in recall: invalid image " + str( image ) )
        return None
    classes = nn_arch[ "classes" ]
    if nn is None:
        session, nn = restore_nn()
    r   = cnn.recall( session, image, nn )
    nr  = 1 + ( r - r.max() ) / r.ptp()
    nr  = nr / nr.sum()
    inx = nr.argsort()
    print( "image classified as   " + classes[ inx[ -1 ] ] + " with probability " + str( nr[ inx[ -1 ] ] ) )
    print( "second possibility is " + classes[ inx[ -2 ] ] + " with probability " + str( nr[ inx[ -2 ] ] ) )
    if label is not None:
        print( "ground truth is " + classes[ int( label ) ] )


"""
# to train:
setup()
train_nn()

# to test using the last checkpoint:
setup()
test_nn()

# to recall on an image using the last checkpoint:
setup()
recall( "cifar_png/table_10.png" )

"""

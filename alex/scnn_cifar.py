"""
A simple convolutional architecture for a classifer on a subset of CIFAR-100
The personalized classes, and the directory with the personalized datasets are
imported from select_cifar, the script that generates the CIFAR-100 selection

the network architecture is defined using TensorFlow slim

Examples of usage:

# training:
setup()
data_set        = read_data_sets()
images, labels  = data_set[ 'train' ]
train_nn( images, labels )

# recall:
setup()
session         = tf.Session()
nn              = restore_nn( session )
recall( session, nn, "cifar_png/plate_01.png" )

# classification recall:
setup()
session         = tf.Session()
nn              = restore_nn( session )
classify( session, nn, "cifar_png/plate_01.png" )

# evaluate:
setup()
data_set        = read_data_sets()
images, labels  = data_set[ 'test' ]
session         = tf.Session()
nn              = restore_nn( session )
acc             = evaluate( session, nn, images, labels )


alex    May 2018
        Jun 2018 - several small adjustments to make the model readable as nengo.TensorNode

"""


import  pickle
import  gzip
import  os
import  numpy
import  sys
import  logging
import  tensorflow              as tf
import  tensorflow.contrib.slim as slim
from    tensorflow.python.framework import dtypes
from    PIL                         import Image

import  select_cifar


"""
main globals
"""
verbose         = 1                 # verbose level
num_epochs      = 1000              # number of learning epochs
learning_rate   = 0.0001            # learning rate
batch_size      = 512               # size to training batches
background      = False             # run in background
log_file        = "scnn_cifar.log"  # logging filename

nn_arch     = {                     # overall network definition
    "color"         : True,         # color images
    "image_size"    : ( 32, 32 ),   # size of the image
    "mask_size"     : ( 5, 5 ),     # linear size of the convolution mask
    "features"      : ( 64, 32 ),   # number of convolution features for each layer
    "pool_size"     : ( 2, 2 ),     # linear size of maxpool windows
    "pool_stride"   : ( 2, 2 ),     # stride of the maxpool windows
    "norm"          : {             # local response normalization parameters
        "depth_radius"  : 5,
        "bias"          : 1.0,
        "alpha"         : 0.001 / 9.0,
        "beta"          : 0.75
    },                              
    "full_layer"    : ( 256, 128 ), # number of neurons in the outer layers
    "classes"       : None,         # classe names
    "n_class"       : None,         # number of classes
}

model_dir   = "./model_dir"


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
    global nn_arch
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


def read_image( image ):
    """
    read an image from file
    """
    if not tf.gfile.Exists( image ):
        tf.logging.fatal( 'File does not exist %s', image )
        return None
    i       = Image.open( image )
    w, h    = i.size
    d       = numpy.array( i.getdata(), dtype=numpy.uint8 )
    img     = norm_images( d )
    return img.reshape( 1, h, w, 3 )


def setup( external=True ):
    """
    set up main variables
    """
    if background:
        verbose = 0
    if external:
        if not import_arch():
            if verbose:
                print "warning: no external neural architecture definition found, using internal"
    setup_cifar()
    if not os.path.isdir( model_dir ):
        os.mkdir( model_dir )
        if verbose:
            print( "model directory " + model_dir + " missing, created" )


def setup_nn( inputs, dropout_probability=0.5 ):
    """
    set up the network using slim calls
    requires as inputs the same kind of inputs (tensors with images) that will be given
    to the netword while training or recalling
    """

    # move this in a more comprehensive sanity check function of nn_arch
    if len( nn_arch[ 'features' ] ) != len( nn_arch[ 'mask_size' ] ):
        if verbose:
            print( "error: mismatch between number of mask sizes and features in nn_arch" )
        return None

    cnt         = 1
    w, h        = nn_arch[ 'image_size' ]
    cls         = nn_arch[ 'n_class' ]
    chn         = 3 if nn_arch[ 'color' ] else 1
    nn          = inputs

    with slim.arg_scope( [ slim.conv2d, slim.fully_connected ],
                      activation_fn         = tf.nn.relu,
                      weights_initializer   = tf.truncated_normal_initializer( stddev=0.1 ) ):
        with slim.arg_scope( [ slim.fully_connected ], biases_initializer=tf.constant_initializer(0.1) ):
            for f, m, p, s, r in zip( nn_arch[ 'features' ], nn_arch[ 'mask_size' ],
                    nn_arch[ 'pool_size' ], nn_arch[ 'pool_stride' ], nn_arch[ 'norm' ] ):
                nn  = slim.conv2d( nn, f, [ m, m], scope=( 'conv_{}'.format( cnt ) ) )
                nn  = slim.max_pool2d( nn, [ p, p ], stride=s, scope=( 'pool_{}'.format( cnt ) ) )
                if r is not None:
                    r[ 'name' ] = 'norm_{}'.format( cnt )
                    nn  = tf.nn.lrn( nn, **r )
                cnt += 1

            nn  = slim.flatten( nn )
            for out in nn_arch[ 'full_layer' ]:
                nn  = slim.fully_connected( nn, out, scope=( 'full_{}'.format( cnt ) ) )
                nn  = slim.dropout( nn, dropout_probability, scope=( 'drop_{}'.format( cnt ) ) )
                cnt += 1

            nn  = slim.fully_connected( nn, cls, biases_initializer=tf.zeros_initializer(), scope="out_layer" )
            nn  = slim.softmax( nn, scope="softmax_out" )

    return nn


def train_nn(
        images,
        labels,
        out_path        = model_dir,
        learning_rate   = learning_rate,
        num_epochs      = num_epochs,
        batch_size      = batch_size,
        save=True ):
    """
    train the network - note that the network definition is included here
    requires as inputs and labels the training set
    most of the function deals with creating a queue of batches feeding the training
    """

    if background:
        logging.basicConfig( filename=log_file, filemode='w', level=logging.INFO )

    n_batch     = images.shape[ 0 ] / batch_size
    capacity    = 5 * batch_size

    with tf.Graph().as_default():
        imgs        = tf.convert_to_tensor( images, dtype=tf.float32 )
        lbls        = tf.convert_to_tensor( labels, dtype=tf.float32 )
        img_init    = tf.placeholder( dtype=imgs.dtype, shape=imgs.shape, name="input" )
        lbl_init    = tf.placeholder( dtype=lbls.dtype, shape=lbls.shape, name="label" )
        img_in      = tf.Variable( img_init, trainable=False, collections=[] )
        lbl_in      = tf.Variable( lbl_init, trainable=False, collections=[] )

        img, lbl    = tf.train.slice_input_producer( [ imgs, lbls ], num_epochs=num_epochs, shuffle=True )
        imgs, lbls  = tf.train.batch( [ img, lbl ], batch_size=batch_size, capacity=capacity )
        nn          = setup_nn( imgs, dropout_probability=0.5 )
        loss        = slim.losses.softmax_cross_entropy( nn, lbls )
        total_loss  = slim.losses.get_total_loss()
        tf.summary.scalar( 'losses/total_loss', total_loss )
        optimizer   = tf.train.AdamOptimizer( learning_rate )
        train_op    = slim.learning.create_train_op( total_loss, optimizer )

        slim.learning.train(
            train_op,
            out_path,
            number_of_steps     = num_epochs * n_batch,
            log_every_n_steps   = 10,
            save_summaries_secs = 1000,
            save_interval_secs  = 1000
        )



def get_chck( model_dir=model_dir, chck_file=None ):
    """
    get full checkpoint filename
    """
    if chck_file is not None:
        chk = os.path.join( model_dir, chck_file )
        if tf.train.checkpoint_exists( chk ):
            return chk
    return tf.train.latest_checkpoint( model_dir )



def restore_nn( session, model_dir=model_dir, chck_file=None ):
    """
    restore the network, using the internal setup to build the architecture,
    and a checkpoint to restore the weights
    """
    w, h    = nn_arch[ 'image_size' ]
    chn     = 3 if nn_arch[ 'color' ] else 1
    x       = tf.placeholder( dtype=tf.float32, shape=( None, w, h, chn ), name="input" )
    y       = setup_nn( x, dropout_probability=1.0 )
    chk     = get_chck( model_dir, chck_file )
    rst     = slim.assign_from_checkpoint_fn( chk, slim.get_model_variables() )
    rst( session )
    return { 'x' : x, 'y' : y }


def print_2class( probabilities ):
    """
    print the top two classifications
    """
    cl  = nn_arch[ 'classes' ]
    inx = probabilities.argsort()
    c1  = cl[ inx[ -1 ] ]
    p1  = probabilities[ inx[ -1 ] ]
    c2  = cl[ inx[ -2 ] ]
    p2  = probabilities[ inx[ -2 ] ]
    print( "image classified as   {:20s} with probability {:6.4f}".format( c1, p1 ) )
    print( "second possibility is {:20s} with probability {:6.4f}".format( c2, p2 ) )


def classify( session, nn, image, label=None ):
    """
    recall the network for a single image, which should be a path name for an image file
    nn is the dictionary with input and output tensors, as returned by restore_nn,
    and print the classification, togehter with the ground truth, if provided
    """
    r   = recall( session, nn, image )
    pr  = 1 + ( r[ 0 ] - r.max() ) / r.ptp()
    pr  = pr / pr.sum()
    if verbose:
        print_2class( pr )
        if label is not None:
            print( "ground truth is " + nn_arch[ 'classes' ][ int( label ) ] )
    return pr


def recall( session, nn, image ):
    """
    recall the network for a single image, which should be a path name for an image file
    nn is the dictionary with input and output tensors, as returned by restore_nn
    """
    if not isinstance( image, str ):
        print( "Error in recall: invalid image " + str( image ) )
        return None
    i       = read_image( image )

    return session.run( nn[ 'y' ], feed_dict={ nn[ 'x' ]: i } )


def evaluate( session, nn, images, labels ):
    """
    evaluate the network over a batch of images and labels
    """
    r       = session.run( nn[ 'y' ], feed_dict={ nn[ 'x' ]: images } )
    pred    = tf.argmax( r, 1 )
    labs    = tf.argmax( labels, 1 )
    acc, op = tf.metrics.accuracy( labs, pred )
    graph   = tf.get_default_graph()
    mvars   = graph.get_collection( 'metric_variables' )
    tf.variables_initializer( mvars ).run( session=session )
    session.run( [ acc, op ] )
    return session.run( [ acc ] )[ 0 ]

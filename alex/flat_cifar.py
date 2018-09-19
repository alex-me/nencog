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

from    tensorflow.contrib.framework    import  get_model_variables
from    tensorflow.contrib.framework    import  get_variables_to_restore

import  select_cifar


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
    "full_layer"    : ( 256, 128 ),     # number of neurons in the fully-connected layers
    "classes"       : None,             # classe names
    "n_class"       : None,             # number of classes
}

channels        = 3
flatten         = 256

model_dir       = "./model_dir"
model_name      = "mycifar"
learning_rate   = 0.001
epochs          = 500
img_formats     = ( "jpg", "png", "gif" )
scopes          = (
        "conv_layer_1",
        "conv_layer_2",
        "full_layer_1",
        "full_layer_2",
        "final_layer"
)


def setup_cifar():
    """
    adapt to the current selection of CIFAR-100
    """
    nn_arch[ "classes" ]    = select_cifar.classes
    nn_arch[ "n_class" ]    = len( select_cifar.classes )


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


def _img_numpy( session, image ):
    """
    read an image as numpy array
    uses tf tools only, so the decoding is limited to the formats supported by tf
    there is no check that the geometry of the image matches with the nn architecture
    """
    if not os.path.exists( image ):
        if verbose:
            print( "Error in _img_numpy: image " + image + " not found" )
        return None
    ext = image.split( '.' )[ -1 ]
    if not ext in img_formats:
        if verbose:
            print( "Error in _img_numpy: image " + image + " not supported" )
        return None
    idata   = tf.gfile.FastGFile( image, 'rb' ).read()
    img     = tf.image.decode_image( idata )
    i       = img.eval( session=session )
    # GIF decoding give shape [ #frames, height, width, 3 ] even for grayscale images
    if ext == 'gif':
        i   = i[ 0 ]
        if not nn_arch[ 'color' ]:
            i   = i.mean( axis=2 )
            i   = i.reshape( i.shape[ 0 ], i.shape[ 1 ], 1 )
    i       = i.astype( numpy.float32 )
    i       = numpy.multiply( i, 1.0 / 255.0 )
    return i


def img_numpy( image ):
    """
    read an image as numpy array with tf tools
    """
    session = start_session( init=False )
    i       = _img_numpy( session, image )
    session.close()
    return i


def set_cnn( x ):
    """
    setup the network, taking the architecture parameters from nn_arch
    this function assumes an arbitrary number of convolutional layers,
    followed by a fully-connected layer, optionally with dropout, and an output layer
    note that the placeholders for input (x) and label (y_) can be provided externally,
    this is useful for queued batch training
    """
    w, h    = nn_arch[ 'image_size' ]
    m       = nn_arch[ "mask_size" ]
    co      = nn_arch[ "features" ]
    k       = nn_arch[ 'pool_size' ]
    s       = nn_arch[ 'pool_stride' ]
    no      = nn_arch[ 'full_layer' ]
    c       = nn_arch[ 'n_class' ]

    # convolutional layers
    x       = tf.layers.conv2d(
            inputs      = x,
            filters     = co[ 0 ],
            kernel_size = m[ 0 ],
            padding     = "same",
            activation  = tf.nn.relu,
            name        = "conv_layer_1"
    )
    x       = tf.layers.max_pooling2d(
            inputs      = x,
            pool_size   = k[ 0 ],
            strides     = s[ 0 ],
            padding     = "same",
            name        = "max_pool_1"
    )
    x       = tf.layers.conv2d(
            inputs      = x,
            filters     = co[ 1 ],
            kernel_size = m[ 1 ],
            padding     = "same",
            activation  = tf.nn.relu,
            name        = "conv_layer_2"
    )
    x       = tf.layers.max_pooling2d(
            inputs      = x,
            pool_size   = k[ 1 ],
            strides     = s[ 1 ],
            padding     = "same",
            name        = "max_pool_2"
    )

    # densely connected layers
    x       = tf.reshape( x, [ -1, flatten ] )
    x       = tf.layers.dense( inputs=x, units=no[ 0 ], activation=tf.nn.relu, name="full_layer_1" )
    x       = tf.layers.dense( inputs=x, units=no[ 1 ], activation=tf.nn.relu, name="full_layer_2" )
    y       = tf.layers.dense( inputs=x, units=c, activation=None, name="final_layer" )

    """
    print "tf.trainable_variables():"
    print tf.trainable_variables()
    print "get_variables_to_restore():"
    print get_variables_to_restore()
    print "get_model_variables():"
    print get_model_variables()
    """

    return y


def train_nn( dset=None, epochs=epochs, learning_rate=0.001, save=True ):
    """
    train the network in batches, checking accuracy on the test set
    note that the batche size is set equal to the test set size, so to use the
    same architecture
    """
    if dset is None:
        dset    = read_data_sets()
    fmt             = "at step %05d accuracy: %g"
    batch           = dset[ 'test' ][ 0 ].shape[ 0 ]
    capacity        = 5 * batch

    with tf.Graph().as_default():
        images      = tf.constant( dset[ 'train' ][ 0 ] )
        labels      = tf.constant( dset[ 'train' ][ 1 ] )
        img, lbl    = tf.train.slice_input_producer( [ images, labels ], num_epochs=epochs, shuffle=True )
        imgs, lbls  = tf.train.batch( [ img, lbl ], batch_size=batch, capacity=capacity )
        y           = set_cnn( imgs )
        ce          = tf.nn.softmax_cross_entropy_with_logits_v2( labels=lbls, logits=y )
        cem         = tf.reduce_mean( ce )
        ts          = tf.train.AdamOptimizer( learning_rate ).minimize( cem )
        session     = start_session( init=True )
        coord       = tf.train.Coordinator()
        queue       = tf.train.start_queue_runners( sess=session, coord=coord )
        i    = 0
        try:
            while not coord.should_stop():
                err, _  = session.run( [ cem, ts ] )
                if verbose:
                    print( fmt % ( i, err ) )
                i   = i + 1
        except tf.errors.OutOfRangeError:
            if verbose:
                print( "end of training" )
            if save:
                chck_file   = os.path.join( model_dir, model_name )
                s           = tf.train.Saver( tf.trainable_variables() )
                if verbose:
                    print "saving trainable variables:"
                    print tf.trainable_variables()
                s.save( session, chck_file, global_step=i )
        finally:
            coord.request_stop()
        coord.join( queue )

    return session


def test_cnn( session, dset, nn ):
    """
    test the network, inputs:
        session     an open tensorflow session descriptor
        dset        dictionary with training and test sets
        nn          network architecture dictionary, as returned by restore_weights
    """
    labels      = tf.constant( dset[ 'test' ][ 1 ] )
    correct     = tf.equal( tf.argmax( nn[ 'y' ], 1 ), tf.argmax( labels, 1 ) )
    accuracy    = tf.reduce_mean( tf.cast( correct, tf.float32 ) )
    feed_dict   = { nn[ 'x' ] : dset[ 'test' ][ 0 ] }
    return session.run( accuracy, feed_dict=feed_dict )


def worst_cnn( session, dset, nn, n=4 ):
    """
    find the labels the most often appear as wrong results
    inputs:
        session     an open tensorflow session descriptor
        dset        couple with images and labels
        nn          network architecture dictionary, as returned by restore_weights
        n           number of wrong labels to output
    """
    out         = ( tf.argmax( nn[ 'y' ], 1 ), tf.argmax(nn[ 'y_' ], 1 ) )
    feed_dict   = {}
    feed_dict[ nn[ 'x' ] ]  = dset[ 0 ]
    feed_dict[ nn[ 'y_' ] ] = dset[ 1 ]
    if 'p' in nn.keys():
        feed_dict[ nn[ 'p' ] ]  = 1.0
    y, y_       = session.run( out, feed_dict=feed_dict )
    boo         = numpy.equal( y, y_ )
    ym          = numpy.ma.array( y_, mask=boo )
    wrong       = ym.compressed()
    u, i        = numpy.unique( wrong, return_counts=True )
    idx         = numpy.flip( i.argsort(), axis=0 )[ : n ]
    return( u[ idx ], i[ idx ] )


def recall( session, image, nn ):
    """
    recall the network, inputs:
        session     an open tensorflow session descriptor
        image       filename of an image, or numpy array
        nn          network architecture dictionary, as returned by restore_weights
    """
    if isinstance( image, numpy.ndarray ):
        img     = image
    else:
        if not isinstance( image, str ):
            if verbose:
                print( "Error in recall: invalid image " + str( image ) )
            return None
        img         = _img_numpy( session, image )

    close       = False
    if session is None:
        session     = start_session()
        close       = True
    feed_dict   = {}
    feed_dict[ nn[ 'x' ] ]  = numpy.array( ( img, ) )
    r           = session.run( nn[ 'y' ], feed_dict=feed_dict )[ 0 ]
    if close:
        session.close()
    return r


def start_session( init=False ):
    s   = tf.Session()
    if init:
        init_op = tf.group( tf.initialize_all_variables(), tf.initialize_local_variables() )
        s.run( init_op )
    return s


def get_chck( chck_file=None ):
    """
    get full checkpoint filename
    """
    if chck_file is not None:
        chk = os.path.join( model_name, chck_file )
        if tf.train.checkpoint_exists( chk ):
            return chk
    return tf.train.latest_checkpoint( model_dir )


def restore_weights( session, chck_file=None ):
    """
    restore weights from the default directory, with the given basename,
    otherwise retrive the last saved weights
    if no network is supplied, initialize it
    """
    w, h    = nn_arch[ 'image_size' ]
    chn     = 3 if nn_arch[ 'color' ] else 1
    x       = tf.placeholder( dtype=tf.float32, shape=( None, w, h, chn ), name="input" )
    y       = set_cnn( x )
    fname   = get_chck( chck_file )
    if fname is None:
        if verbose:
            print( "Error in restore_weights: no valid checkpoint found" )
        return None
    s       = tf.train.Saver( tf.trainable_variables() )
    s.restore( session, fname )
    return { 'x' : x, 'y' : y }


def setup():
    """
    set up main variables
    NOTE that the sequence of expressions is crucial, in order to preserve a valid nn_arch
    and export it correctly to cnn
    """
    setup_cifar()
    if not os.path.isdir( model_dir ):
        os.mkdir( model_dir )
        if verbose:
            print( "model directory " + model_dir + " missing, created" )



def restore_nn( session=None ):
    if session is None:
        session = start_session()
    return session, restore_weights( session )


def test_nn( data_set=None):
    if data_set is None:
        data_set    = read_data_sets()
    session, nn = restore_nn()
    t           = test_cnn( session, data_set, nn )
    session.close()
    return t


def classify( image, session=None, nn=None, label=None ):
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
    r   = recall( session, image, nn )
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
classify( "cifar_png/table_10.png" )

"""

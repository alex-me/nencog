"""
wrapper around tensorflow primitives for defining CNN's
and perform training

- derived from nencog/alex/my_cifar/cnn.py

alex    Nov 2018
"""


import  os
import  re
import  numpy
import  tensorflow   as tf

"""
main globals
"""
verbose     = 1                         # verbose level
nn_arch     = None                      # defined elsewere
weights_dir = None                      # directory where weights are stored
model_name  = None                      # name of the model in checkpoint files
chck_pre    = "chck_"                   # prefix of latest_filename argument in tf.train.latest_checkpoint
img_formats = ( "jpg", "png", "gif" )   # supported image formats


def init_weights( shape, stddev=0.1 ):
    w   = lambda: tf.truncated_normal( shape, stddev=stddev )
    return tf.Variable( initial_value=w, name="weights" )


def init_bias( shape, value=0.1 ):
    b   = lambda: tf.constant( value, shape=shape )
    return tf.Variable( initial_value=b, name="bias" )


def cnn( x, W, b, strides=[1, 1, 1, 1] ):
    return tf.nn.relu( tf.nn.conv2d( x, W, strides=strides, padding='SAME' ) + b )


def mpool_norm( x, layer ):
    k       = nn_arch[ 'pool_size' ][ layer ]
    s       = nn_arch[ 'pool_stride' ][ layer ]
    ksize   = [ 1, k, k, 1 ]
    strides = [ 1, s, s, 1 ]
    if nn_arch[ 'norm' ][ layer ] is not None:
        x   = tf.nn.lrn( x, **nn_arch[ 'norm' ][ layer ] )
    return tf.nn.max_pool( x, ksize=ksize, strides=strides, padding='SAME' )
        

def conv_layer( x, layer ):
    """
    define one convolutional layer, using architecture parameters in nn_arch
    """
    channels    = 3     if nn_arch[ 'color' ]   else 1
    m   = nn_arch[ 'mask_size' ][ layer ]
    fi  = nn_arch[ 'features' ][ layer - 1 ]    if layer    else channels
    fo  = nn_arch[ 'features' ][ layer ]
    W   = init_weights( [ m, m, fi, fo ] )
    b   = init_bias( [ fo ] )
    return cnn( x, W, b )
        

def full_layer( x, layer, dropout_probability=None ):
    """
    define one fully connected layer with dropout
    if this is the first layer after the convolutional layers, takes care
    of reshaping the input as tensor in one dimention, taking into account
    the initial image size, and the number of pooling operations done
    """
    l_name  = "full_layer_{}".format( layer )
    o   = nn_arch[ 'full_layer' ][ layer ]
    if layer > 0:
        n   = nn_arch[ 'full_layer' ][ layer - 1 ]
        x1  = x
    else:
        p       = numpy.array( nn_arch[ 'pool_stride' ] )
        w, h    = nn_arch[ 'image_size' ]
        d       = p.prod()
        # this computation allow use of image sizes that are not exact multiple of d
        h       = round( float( h ) / d )
        w       = round( float( w ) / d )
        n       = int( h * w ) * nn_arch[ 'features' ][ -1 ]
        x1      = tf.reshape( x, [ -1, n ] )
    with tf.variable_scope( l_name ) as scope:
        W   = init_weights( [ n, o ] )
        b   = init_bias( [ o ] )
        y   = tf.nn.relu( tf.matmul( x1, W ) + b, name="out" )
    if dropout_probability is not None:
        return tf.nn.dropout( y, dropout_probability )
    return y
        

def final_layer( x ):
    """
    define the final output layer
    """
    o   = nn_arch[ 'full_layer' ][ -1 ]
    c   = nn_arch[ 'n_class' ]
    with tf.variable_scope( "final_layer" ) as scope:
        W   = init_weights( [ o, c ] )
        b   = init_bias( [ c ] )
    return tf.matmul( x, W ) + b
        

def set_cnn( x ):
    """
    setup the network, taking the architecture parameters from nn_arch
    this function assumes an arbitrary number of convolutional layers,
    followed by a fully-connected layer, optionally with dropout, and an output layer
    note that the placeholders for input (x) and label (y_) can be provided externally,
    this is useful for queued batch training
    """
    w, h        = nn_arch[ 'image_size' ]
    channels    = 3     if nn_arch[ 'color' ]   else 1
    if len( nn_arch[ 'features' ] ) != len( nn_arch[ 'mask_size' ] ):
        if verbose:
            print( "error: mismatch between number of mask sizes and features in nn_arch" )
        return None
    c   = nn_arch[ 'n_class' ]

    # convolutional layers
    for l in range( len( nn_arch[ 'features' ] ) ):
        l_name  = "conv_layer_{}".format( l )
        with tf.variable_scope( l_name ) as scope:
            x   = conv_layer( x, l )
            x   = mpool_norm( x, l )

    # densely connected layers, optionally with dropout
    for l in range( len( nn_arch[ 'full_layer' ] ) ):
        x   = full_layer( x, l )

    # layer 4 - readout
    y   = final_layer( x )

    return y


def train_cnn( dset, epochs=100, learning_rate=0.001, save=True ):
    """
    train the network in batches, checking accuracy on the test set
    note that the batche size is set equal to the test set size, so to use the
    same architecture
    saving is done on weights_dir, using model_name as model name, and a personalized
    latest_filename (instead of the default "checkpoint")
    """
    fmt             = "at step %05d accuracy: %g"
    batch           = dset[ 'test' ][ 0 ].shape[ 0 ]
    capacity        = 5 * batch
    all_imgs        = dset[ 'train' ][ 0 ]
    all_lbls        = dset[ 'train' ][ 1 ]

    with tf.Graph().as_default():
        img_place   = tf.placeholder( dtype=all_imgs.dtype, shape=all_imgs.shape )
        lbl_place   = tf.placeholder( dtype=all_lbls.dtype, shape=all_lbls.shape )
        img_in      = tf.Variable( img_place, trainable=False, collections=[] )
        lbl_in      = tf.Variable( lbl_place, trainable=False, collections=[] )

        img, lbl    = tf.train.slice_input_producer( [ img_in, lbl_in ], num_epochs=epochs, shuffle=True )
        imgs, lbls  = tf.train.batch( [ img, lbl ], batch_size=batch, capacity=capacity )
        y           = set_cnn( imgs )
        ce          = tf.nn.softmax_cross_entropy_with_logits_v2( labels=lbls, logits=y )
        cem         = tf.reduce_mean( ce )
        ts          = tf.train.AdamOptimizer( learning_rate ).minimize( cem )

        session     = start_session( init=True )
        session.run( img_in.initializer, feed_dict={img_place: all_imgs} )
        session.run( lbl_in.initializer, feed_dict={lbl_place: all_lbls} )
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
                chck_file       = os.path.join( weights_dir, model_name )
                latest_filename = chck_pre + model_name
                s               = tf.train.Saver( tf.trainable_variables() )
                s.save( session, chck_file, global_step=i, latest_filename=latest_filename )
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


def recall( session, images, nn ):
    """
    recall the network, inputs:
        session     an open tensorflow session descriptor
        image       single image as numpy array, or batch of images
        nn          network architecture dictionary, as returned by restore_weights
    """
    if not isinstance( images, numpy.ndarray ):
        if verbose:
            print( "Error in recall: invalid image format" )
        return None

    single_image    = False
    if len( images.shape ) == 3:
        images          = numpy.array( ( images, ) )
        single_image    = True
    close       = False
    if session is None:
        session     = start_session()
        close       = True
    feed_dict   = { nn[ 'x' ] : images }
    r           = session.run( nn[ 'y' ], feed_dict=feed_dict )
    if close:
        session.close()
    if single_image:
        r   = r[ 0 ]
    return r


def start_session( init=False, graph=None ):
    s   = tf.Session( graph=graph )
    if init:
        init_op = tf.group( tf.initialize_all_variables(), tf.initialize_local_variables() )
        s.run( init_op )
    return s


def get_chck( chck_file=None ):
    """
    get full checkpoint filename
    """
    if chck_file is not None:
        chk = os.path.join( weights_dir, chck_file )
        if tf.train.checkpoint_exists( chk ):
            return chk
    latest_filename = chck_pre + model_name
    return tf.train.latest_checkpoint( weights_dir, latest_filename=latest_filename )


def restore_weights( session, chck_file=None ):
    """
    restore weights from the default directory, with the given basename,
    otherwise retrive the last saved weights
    if no network is supplied, initialize it
    """
    fname   = get_chck( chck_file )
    if fname is None:
        if verbose:
            print( "Error in restore_weights: no valid checkpoint found" )
        return None
    w, h    = nn_arch[ 'image_size' ]
    chn     = 3 if nn_arch[ 'color' ] else 1
    with session.as_default():
        x       = tf.placeholder( dtype=tf.float32, shape=( None, h, w, chn ), name="input" )
        y       = set_cnn( x )
        s       = tf.train.Saver( tf.trainable_variables() )
        s.restore( session, fname )
    return { 'x' : x, 'y' : y }

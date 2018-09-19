"""
wrapper around tensorflow primitives for defining CNN's
and perform training

alex    May 2018
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
img_formats = ( "jpg", "png", "gif" )   # supported image formats


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


def init_weights( shape, stddev=0.1 ):
    w   = tf.truncated_normal( shape, stddev=stddev )
    return tf.Variable( w, name="weights" )


def init_bias( shape, value=0.1 ):
    b   = tf.constant( value, shape=shape )
    return tf.Variable( b, name="bias" )


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
        

def set_cnn( x=None, y_=None, learning_rate=0.0001 ):
    """
    setup the network, taking the architecture parameters from nn_arch
    this function assumes an arbitrary number of convolutional layers,
    followed by a fully-connected layer, optionally with dropout, and an output layer
    note that the placeholders for input (x) and label (y_) can be provided externally,
    this is useful for queued batch training
    """
    w, h        = nn_arch[ 'image_size' ]
    channels    = 3     if nn_arch[ 'color' ]   else 1
    drop_prob   = False
    # move this in a more comprehensive sanity check function of nn_arch
    if len( nn_arch[ 'features' ] ) != len( nn_arch[ 'mask_size' ] ):
        if verbose:
            print( "error: mismatch between number of mask sizes and features in nn_arch" )
        return None
    c   = nn_arch[ 'n_class' ]
    if x is None:
        x   = tf.placeholder( tf.float32, shape=[ None, w, h, channels ], name="input_image" )
    if y_ is None:
        y_  = tf.placeholder( tf.float32, [ None, c ], name="label" )
    p   = tf.placeholder( tf.float32, name="dropout_probability" )

    # convolutional layers
    x1  = x
    for l in range( len( nn_arch[ 'features' ] ) ):
        l_name  = "conv_layer_{}".format( l )
        with tf.variable_scope( l_name ) as scope:
            y   = conv_layer( x1, l )
            x1  = mpool_norm( y, l )

    # densely connected layers, optionally with dropout
    for l in range( len( nn_arch[ 'full_layer' ] ) ):
        if nn_arch[ 'use_dropout' ][ l ]:
            x1          = full_layer( x1, l, p )
            drop_prob   = True
        else:
            x1          = full_layer( x1, l )

    # layer 4 - readout
    y   = final_layer( x1 )

    # learning rule
    ce  = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( labels=y_, logits=y  ) )
    ts  = tf.train.AdamOptimizer( learning_rate ).minimize( ce )

    # include a saver
    s   = tf.train.Saver()

    nn  = {}
    nn[ 'x' ]   = x
    nn[ 'y' ]   = y
    nn[ 'y_' ]  = y_
    nn[ 'ts' ]  = ts
    if drop_prob:
        nn[ 'p' ]   = p
    nn[ 's' ]   = s
    return nn


def train_cnn( dset, epochs=100, learning_rate=0.001, save=True ):
    """
    train the network in batches, checking accuracy on the test set
    note that the batche size is set equal to the test set size, so to use the
    same architecture
    """
    fmt             = "at step %05d accuracy: %g"
    batch           = dset[ 'test' ][ 0 ].shape[ 0 ]
    capacity        = 5 * batch

    with tf.Graph().as_default():
        images      = tf.constant( dset[ 'train' ][ 0 ] )
        labels      = tf.constant( dset[ 'train' ][ 1 ] )
        img, lbl    = tf.train.slice_input_producer( [ images, labels ], num_epochs=epochs, shuffle=True )
        imgs, lbls  = tf.train.batch( [ img, lbl ], batch_size=batch, capacity=capacity )
        nn          = set_cnn( imgs, lbls, learning_rate=learning_rate )
        session     = start_session( init=True )
        coord       = tf.train.Coordinator()
        queue       = tf.train.start_queue_runners( sess=session, coord=coord )
        i    = 0
        try:
            while not coord.should_stop():
                if verbose:
                    t       = test_cnn( session, dset, nn )
                    print( fmt % ( i, t ) )
                session.run( nn[ 'ts' ] )
                i   = i + 1
        except tf.errors.OutOfRangeError:
            if verbose:
                print( "end of training" )
            if save:
                chck_file   = os.path.join( weights_dir, model_name )
                nn[ 's' ].save( session, chck_file, global_step=i )
        finally:
            coord.request_stop()
        coord.join( queue )
        session.close()

    return nn


def test_cnn( session, dset, nn ):
    """
    test the network, inputs:
        session     an open tensorflow session descriptor
        dset        dictionary with training and test sets
        nn          network architecture dictionary, as returned by set_cnn
    """
    correct     = tf.equal( tf.argmax( nn[ 'y' ], 1 ), tf.argmax(nn[ 'y_' ], 1 ) )
    accuracy    = tf.reduce_mean( tf.cast( correct, tf.float32 ) )
    feed_dict   = {}
    feed_dict[ nn[ 'x' ] ]  = dset[ 'test' ][ 0 ]
    feed_dict[ nn[ 'y_' ] ] = dset[ 'test' ][ 1 ]
    if 'p' in nn.keys():
        feed_dict[ nn[ 'p' ] ]  = 1.0
    return session.run( accuracy, feed_dict=feed_dict )


def worst_cnn( session, dset, nn, n=4 ):
    """
    find the labels the most often appear as wrong results
    inputs:
        session     an open tensorflow session descriptor
        dset        couple with images and labels
        nn          network architecture dictionary, as returned by set_cnn
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
        nn          network architecture dictionary, as returned by set_cnn
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
    if 'p' in nn.keys():
        feed_dict[ nn[ 'p' ] ]  = 1.0
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
        chk = os.path.join( weights_dir, chck_file )
        if tf.train.checkpoint_exists( chk ):
            return chk
    return tf.train.latest_checkpoint( weights_dir )


def restore_weights( session, nn=None, chck_file=None ):
    """
    restore weights from the default directory, with the given basename,
    otherwise retrive the last saved weights
    if no network is supplied, initialize it
    """
    if nn is None:
        nn = set_cnn()
    fname   = get_chck( chck_file )
    if fname is None:
        if verbose:
            print( "Error in restore_weights: no valid checkpoint found" )
        return None
    nn[ 's' ].restore( session, fname )
    return nn

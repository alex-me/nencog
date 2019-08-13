"""
#############################################################################################################

Functions for training a model for LAVA object classification

    alex   2019

#############################################################################################################
"""

import  os
import  sys
import  datetime
import  pickle
import  gzip

import  numpy                               as np
from    math                import ceil, sqrt, inf
from    keras               import utils, models, callbacks, preprocessing
from    keras               import backend  as K

import  matplotlib
matplotlib.use( 'agg' )     # to use matplotlib with unknown 'DISPLAY' var (when using remote display)
from    matplotlib          import pyplot   as plt

import  mesg                                as ms
import  pack_lava                           as pl


SHUFFLE         = True

cnfg            = []        # NOTE initialized by 'nn_main.py'

dir_check       = 'chkpnt'
nn_best         = 'nn_best.h5'

file_ext        = ( '.png', '.jpg', '.jpeg' )
data_ext        = '.gz'
cond_ext        = lambda x: x.lower().endswith( file_ext )      # the condition an image file must satisfy
cond_mem        = lambda x: x.lower().endswith( data_ext )      # the condition of dataset on memory


def iter_simple( dr, shuffle=True ):
    """ -----------------------------------------------------------------------------------------------------
    Simple generic dataset iterator

    https://keras.io/preprocessing/image/#imagedatagenerator-methods

    dr:             [str] folder of dataset (it must contain a subfolder for each class)
    shuffle:        [bool] whether to shuffle the data

    return:         [keras_preprocessing.image.DirectoryIterator]
    ----------------------------------------------------------------------------------------------------- """
    
    # 'rescale' to normalize pixels in [0..1]
    idg     = preprocessing.image.ImageDataGenerator( rescale=1./255 )

    flow    = idg.flow_from_directory(
            directory   = dr,
            target_size = cnfg[ 'img_size' ][ :-1 ],
            color_mode  = 'rgb',
            class_mode  = 'categorical',
            classes     = [ 'yes', 'no' ],
            batch_size  = cnfg[ 'batch_size' ],
            shuffle     = shuffle,
            seed        = cnfg[ 'seed' ]
    )

    return flow


def gen_dataset( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Iterate over a training and a validation set, where target images are equal to input images

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)

    return:         [list] of keras_preprocessing.image.DirectoryIterator
    ----------------------------------------------------------------------------------------------------- """
    train_dir   = os.path.join( dir_dset, 'train' )
    valid_dir   = os.path.join( dir_dset, 'valid' )

    train_flow  = iter_simple( train_dir, shuffle=True )
    valid_flow  = iter_simple( valid_dir, shuffle=True )

    return train_flow, valid_flow


def len_dataset( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Return the number of samples in each subset (train/valid/test) of a dataset

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)

    return:         [list of int]
    ----------------------------------------------------------------------------------------------------- """
    train   = None
    valid   = None
    test    = None

    for dirpath, dirname, filename in os.walk( dir_dset ):
        if not dirname:
            if "train" in dirpath:
                train   = dirpath
            elif "valid" in dirpath:
                valid   = dirpath
            elif "test" in dirpath:
                test    = dirpath

    tr  = len( [ f for f in os.listdir( train ) if cond_ext( f ) ] )
    vl  = len( [ f for f in os.listdir( valid ) if cond_ext( f ) ] )
    ts  = len( [ f for f in os.listdir( test ) if cond_ext( f ) ] )

    return tr, vl, ts


def split_valid( data, n ):
    """ -----------------------------------------------------------------------------------------------------
    split data into train and valid sets

    n:          [int] number of validation data (should be even)
    ----------------------------------------------------------------------------------------------------- """
    n_2     = n // 2
    valid_y = data[ : n_2 ]
    valid_n = data[ - n_2 : ]
    valid   = np.concatenate( ( valid_y, valid_n ) )
    train   = data[ n_2 : - n_2 ]
    return { "valid" : valid, "train" : train }


def read_data_set():
    """
    read the dataset of a object category, split into train and valid, and
    reorganize into the format used by cnn
    """
    f       = cnfg[ 'dir_dset' ]
    gf      = gzip.open( f )
    dset    = pickle.load( gf )
    gf.close()

    imgs    = dset[ "images" ]
    labs    = dset[ "labels" ]
    n       = imgs.shape[ 0 ]
    n_valid = pl.divisor * round( 0.05 * ( n / pl.divisor ) )
    i_split = split_valid( imgs, n_valid )
    l_split = split_valid( labs, n_valid )
    train   = i_split[ "train" ], l_split[ "train" ]
    valid   = i_split[ "valid" ], l_split[ "valid" ]

    return train, valid


def set_callback():
    """ -----------------------------------------------------------------------------------------------------
    Set of functions to call during the training procedure
        - save checkpoint of the best model so far
        - save 'n_check' checkpoints during training
        - optionally save information for TensorBoard
        - end training after 'patience' unsuccessful epochs

    return:         [keras.callbacks.ModelCheckpoint]
    ----------------------------------------------------------------------------------------------------- """
    calls   = []

    if cnfg[ 'n_check' ] != 0:
        period  = ceil( cnfg[ 'n_epochs' ] / cnfg[ 'n_check' ] )

        if cnfg[ 'n_check' ] > 0:
            calls.append( callbacks.ModelCheckpoint(
                    os.path.join( cnfg[ 'dir_current' ], nn_best ),
                    save_best_only          = True,
                    save_weights_only       = False,
                    period                  = 1
            ) )

        if cnfg[ 'n_check' ] > 1:
            p       = os.path.join( cnfg[ 'dir_current' ], dir_check )
            fname   = os.path.join( p, "check_{epoch:04d}.h5" )
            os.makedirs( p )
            calls.append( callbacks.ModelCheckpoint(
                        fname,
                        save_weights_only   = False,
                        period              = period
            ) )

    if cnfg[ 'tboard' ]:
        calls.append( callbacks.TensorBoard(
                    log_dir                 = cnfg[ 'dir_current' ],
                    histogram_freq          = 0,
                    batch_size              = 1,
                    write_graph             = True,
                    write_grads             = False,
                    write_images            = True
        ) )

    if cnfg[ 'patience' ] > 0:
        calls.append( callbacks.EarlyStopping( monitor='val_loss', patience=cnfg[ 'patience' ] ) )

    return calls



def train_from_disk( model ):
    """ -----------------------------------------------------------------------------------------------------
    Training procedure with data from disk

    model:          [keras.engine.training.Model]

    return:         [keras.callbacks.History]
    ----------------------------------------------------------------------------------------------------- """

    # train and valid dataset generators
    train_feed, valid_feed                      = gen_dataset( cnfg[ 'dir_dset' ] )
    train_samples, valid_samples, test_samples  = len_dataset( cnfg[ 'dir_dset' ] )
    steps_per_epoch                             = ceil( train_samples / cnfg[ 'batch_size' ] )
    validation_steps                            = ceil( valid_samples / cnfg[ 'batch_size' ] )

    history                 = model.fit_generator(
            train_feed,
            epochs              = cnfg[ 'n_epochs' ],
            validation_data     = valid_feed,
            steps_per_epoch     = steps_per_epoch,
            validation_steps    = validation_steps,
            callbacks           = set_callback(),
            verbose             = 2,
            shuffle             = SHUFFLE
    )

    return history


def train_from_memory( model ):
    """ -----------------------------------------------------------------------------------------------------
    Training procedure with data in memory

    model:          [keras.engine.training.Model]

    return:         [keras.callbacks.History]
    ----------------------------------------------------------------------------------------------------- """
    train_set, valid_set    = read_data_set()
    train_samples           = len( train_set[ 0 ] )
    valid_samples           = len( valid_set[ 0 ] )

    history                 = model.fit(
            x                   = train_set[ 0 ],
            y                   = train_set[ 1 ],
            batch_size          = cnfg[ 'batch_size' ],
            epochs              = cnfg[ 'n_epochs' ],
            validation_data     = valid_set,
            callbacks           = set_callback(),
            verbose             = 2,
            shuffle             = SHUFFLE
    )

    return history


def train_model( model, tlog ):
    """ -----------------------------------------------------------------------------------------------------
    Training procedure, using datasets that can be loaded in memory, or flowing files from disk

    NOTE: the choice between memory/disk is simply inferred from the name of the dataset: if
    it has the extension of a compressed file it is loaded in memory, otherwise it is assumed
    as the root directory of a dataset tree that adhers  to Keras conventions

    model:          [keras.engine.training.Model]
    tlog:           [str] path to log file

    return:         [keras.callbacks.History]
    ----------------------------------------------------------------------------------------------------- """
    t_start = datetime.datetime.now()       # starting time of execution
    if cond_mem( cnfg[ 'dir_dset' ] ):
        history     = train_from_memory( model )
    else:
        history     = train_from_disk( model )

    t_end   = datetime.datetime.now()                           # end time of execution

    with open( tlog, 'w' ) as f:
        f.write( "Training time:\n" )
        f.write( str( t_end - t_start ) + '\n' )                # save total time of execution

    return history



def plot_history( history, fname='loss' ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the loss performance during training

    history:        [keras.callbacks.History]
    fname:          [str] name of output file without extension
    ----------------------------------------------------------------------------------------------------- """
    train_loss  = history.history[ 'loss' ]
    valid_loss  = history.history[ 'val_loss' ]
    epochs      = range( 1, len( train_loss ) + 1 )

    plt.plot( epochs, train_loss, 'r--' )
    plt.plot( epochs, valid_loss, 'b-' )
    plt.legend( [ 'Training Loss', 'Validation Loss' ] )
    plt.xlabel( 'Epoch' )
    plt.ylabel( 'Loss' )
    plt.savefig( "{}.pdf".format( fname ) )

    if len( train_loss ) > 5:
        m   = np.mean( train_loss )
        s   = np.std( train_loss )
        plt.ylim( [ m - s, m + s ] )
        plt.savefig( "{}_zoom.pdf".format( fname ) )

    plt.close()

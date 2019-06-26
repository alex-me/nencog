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
    read the dataset of a object category, split into teain and valid, and
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



def train_model( model, tlog ):
    """ -----------------------------------------------------------------------------------------------------
    Training procedure for model predicting in time

    model:          [keras.engine.training.Model]
    tlog:           [str] path to log file

    return:         [keras.callbacks.History]
    ----------------------------------------------------------------------------------------------------- """
    train_set, valid_set    = read_data_set()
    train_samples           = len( train_set[ 0 ] )
    valid_samples           = len( valid_set[ 0 ] )

    t_start                 = datetime.datetime.now()       # starting time of execution

    history                 = model.fit(
            x                   = train_set[ 0 ],
            y                   = train_set[ 1 ],
            batch_size          = cnfg[ 'batch_size' ],
            epochs              = cnfg[ 'n_epochs' ],
            validation_data     = valid_set,
#           steps_per_epoch     = train_samples // cnfg[ 'batch_size' ],
#           validation_steps    = valid_samples // cnfg[ 'batch_size' ],
            callbacks           = set_callback(),
            verbose             = 2,
            shuffle             = SHUFFLE
    )

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

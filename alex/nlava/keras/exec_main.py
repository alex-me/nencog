"""
#############################################################################################################

neural network models for LAVA object classification

    alex   2019

#############################################################################################################
"""

import  numpy                       as np
import  random                      as rn
import  cnfg                        as cf

# NOTE seed must be set before importing anything from Keras or TF
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
if __name__ == '__main__':
    args                = cf.get_args()
    cnfg                = cf.get_config( args[ 'CONFIG' ] )
    np.random.seed( cnfg[ 'seed' ] )
    rn.seed( cnfg[ 'seed' ] )

import  os
import  sys
import  time

import  tensorflow                  as tf
from    keras       import backend  as K

import  mesg                        as ms
import  arch                        as ar
import  trainer                     as tr
import  pack_lava                   as pl


SAVE                = False
FRMT                = "%y-%m-%d_%H-%M-%S"

dir_res             = 'res'
dir_log             = 'log'
dir_eval            = 'eval'
dir_cnfg            = 'config'
dir_src             = 'src'

log_err             = os.path.join( dir_log, "log.err" )
log_out             = os.path.join( dir_log, "log.out" )
log_msg             = os.path.join( dir_log, "log.msg" )
log_time            = os.path.join( dir_log, "log.time" )

nn                  = None
dir_current         = None



def init_config():
    """ -----------------------------------------------------------------------------------------------------
    Initialization
    ----------------------------------------------------------------------------------------------------- """
    global dir_current

    dir_current     = os.path.join( dir_res, time.strftime( FRMT ) )
    os.makedirs( dir_current )
    os.makedirs( os.path.join( dir_current, dir_log ) )

    # redirect stderr and stdout in log files
    if args[ 'REDIRECT' ]:
        le          = os.path.join( dir_current, log_err )
        lo          = os.path.join( dir_current, log_out )
        sys.stderr  = open( le, 'w' )
        sys.stdout  = open( lo, 'w' )

        # how to restore
        # sys.stdout = sys.__stdout__

    #exec( "from {} import cnfg".format( args[ 'CONFIG' ] ) )

    cnfg[ 'dir_current' ]   = dir_current
    cnfg[ 'log_msg' ]       = os.path.join( dir_current, log_msg )

    # visible GPUs - must be here, before all the keras stuff
    n_gpus  = eval( args[ 'GPU' ] )
    if isinstance( n_gpus, int ):
        os.environ[ "CUDA_VISIBLE_DEVICES" ]    = str( list( range( n_gpus ) ) )[ 1 : -1 ]
    elif isinstance( n_gpus, ( tuple, list ) ):
        os.environ[ "CUDA_VISIBLE_DEVICES" ]    = str( n_gpus )[ 1 : -1 ]
        n_gpus                                  = len( n_gpus )
    else:
       ms.print_err( "GPUs specification {} not valid".format( n_gpus ) ) 
    cnfg[ 'n_gpus' ]    = n_gpus

    # GPU memory fraction
    if n_gpus > 0:
        tf.set_random_seed( cnfg[ 'seed' ] )
        tf_cnfg                                             = tf.ConfigProto()
        tf_cnfg.gpu_options.per_process_gpu_memory_fraction = args[ 'FGPU' ]
        tf_session                                          = tf.Session( config=tf_cnfg )
        K.set_session( tf_session )

    if not cnfg[ 'data_class' ] in pl.categories:
       ms.print_err( "Category {} not valid".format( cnfg[ 'data_class' ] ) ) 

    # specialize image shape according to data category
    h, w                = pl.sizes[ cnfg[ 'data_class' ] ]
    cnfg[ 'img_size' ]  = ( w, h, 3 )

    # share globals
    ar.cnfg     = cnfg
    tr.cnfg     = cnfg



def create_model():
    """ -----------------------------------------------------------------------------------------------------
    Model architecture and weight training
    ----------------------------------------------------------------------------------------------------- """
    global nn, test_set, test_str

    ar.TRAIN    = args[ 'TRAIN' ]
    nn          = ar.create_model()

    if args[ 'LOAD' ] is not None:
        if not ar.load_model( nn, args[ 'LOAD' ] ):
           ms.print_err( "Failed to load weights from {}".format( args[ 'LOAD' ] ) ) 

    if args[ 'TRAIN' ]:
        lg  = os.path.join( dir_current, log_time )
        hs  = tr.train_model( nn, lg )
        tr.plot_history( hs, os.path.join( dir_current, 'loss' ) )

        if SAVE:
            ar.save_model( nn )


def archive():
    """ -----------------------------------------------------------------------------------------------------
    Archiving
    ----------------------------------------------------------------------------------------------------- """
    # save config files
    if args[ 'ARCHIVE' ] >= 1:
        d       = os.path.join( dir_current, dir_cnfg )
        cfile   =  args[ 'CONFIG' ]
        os.makedirs( d )
        os.system( "cp {} {}".format( cfile, d ) )

    # save python sources
    if args[ 'ARCHIVE' ] >= 2:
        d       = os.path.join( dir_current, dir_src )
        pfile   = "src/*.py"
        os.makedirs( d )
        os.system( "cp {} {}".format( pfile, d ) )


# ===========================================================================================================

if __name__ == '__main__':
    init_config()
    if args[ 'ARCHIVE' ] > 0:
        archive()
    create_model()

# ===========================================================================================================

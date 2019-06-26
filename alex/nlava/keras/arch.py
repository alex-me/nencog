"""
#############################################################################################################

Definition of the neural network architecture

    alex   2019

#############################################################################################################
"""

import  os
import  re
import  numpy               as np
import  tensorflow          as tf

from    abc                 import ABC, abstractmethod
from    keras               import models, layers, utils, initializers, regularizers, optimizers
from    keras               import metrics, losses, preprocessing
from    keras               import backend      as K

import  mesg                        as ms


PLOT            = True
TRAIN           = True

nn_wght         = 'nn_wght.h5'          # filename of model weights
nn_arch         = 'nn_arch.json'        # filename of model architecture
dir_model       = 'models'              # folder containing pre-trained models
dir_plot        = 'plot'                # folder containing the architecture plots
plot_ext        = '.png'                # file format of the plot (pdf/png/jpg)

cnfg            = []                    # NOTE initialized by 'nn_main.py'

layer_code      = {                     # codes specifying the type of layer, passed to 'cnfg.py'
        'conv'  : 'C',
        'dcnv'  : 'T',
        'dnse'  : 'D',
        'flat'  : 'F',
        'pool'  : 'P',
        'rshp'  : 'R',
        'stop'  : '-'
}


# ===========================================================================================================
#
#   - get_optim
#   - model_summary
#
#   - load_model
#   - save_model
#   - create_model
#
# ===========================================================================================================


def get_optim():
    """ -------------------------------------------------------------------------------------------------
    Return an Optimizer according to the object attribute

    return:         [keras.optimizers.Optimizer]
    ------------------------------------------------------------------------------------------------- """
    if cnfg[ 'optimizer' ] == 'ADAGRAD':
        return optimizers.Adagrad( lr=cnfg[ 'lrate' ] )
    if cnfg[ 'optimizer' ] == 'SDG':
        return optimizers.SGD( lr=cnfg[ 'lrate' ] )
    if cnfg[ 'optimizer' ] == 'RMS':
        return optimizers.RMSprop( lr=cnfg[ 'lrate' ] )
    if cnfg[ 'optimizer' ] == 'ADAM':
        return optimizers.Adam( lr=cnfg[ 'lrate' ] )
    
    ms.print_err( "Optimizer {} not valid".format( cnfg[ 'optimizer' ] ) )
        

    
def model_summary( model, fname='model' ):
    """ -------------------------------------------------------------------------------------------------
    Print a summary of the model, and plot a graph of the model and save it to a file

    model:          [keras.engine.training.Model]
    fname:          [str] name of the output image without extension
    ------------------------------------------------------------------------------------------------- """
    if PLOT:
        utils.print_summary( model )

        d   = os.path.join( cnfg[ 'dir_current' ], dir_plot )
        if not os.path.exists( d ):
            os.makedirs( d )

        f   = os.path.join( d, fname + plot_ext )

        #utils.plot_model( model, to_file=f, show_shapes=True, show_layer_names=True, expand_nested=True )
        utils.plot_model( model, to_file=f, show_shapes=True, show_layer_names=True )



def load_model( model, ref_model ):
    """ -----------------------------------------------------------------------------------------------------
    Initialize the weigths of current model usign the weigths of another reference model.
    The two models should have compatible architectures.

    If a folder is passed, it should contain the two files HDF5 and JSON.
    If a single file is passed, it is considered as model+weights HDF5 file

    model:          [keras.engine.training.Model] current model
    ref_model:      [str] folder or filename of the reference model

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    if ref_model.endswith( '.h5' ):         # single file
        h5  = ref_model
    else:                                   # folder
        h5  = os.path.join( ref_model, nn_wght )

    return hl.load_h5( model, h5 )

    
def save_model( model ):
    """ -----------------------------------------------------------------------------------------------------
    Save a trained model in two files: one file for the architecture (JSON) and one for the weights (HDF5)

    model:          [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    model.save_weights( os.path.join( cnfg[ 'dir_current' ], nn_wght ) )

    with open( os.path.join( cnfg[ 'dir_current' ], nn_arch ), 'w' ) as f:
        f.write( model.to_json() )



def create_model():
    """ -----------------------------------------------------------------------------------------------------
    Create the model of the neural network

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    nn  = Classifier()

    nn.model.compile(
            optimizer       = get_optim(),
            loss            = losses.categorical_crossentropy
    )

    return nn.model



# ===========================================================================================================
#
#   Classes
#
#
# ===========================================================================================================

class Classifier( ABC ):
    """ -----------------------------------------------------------------------------------------------------
    class for the generation of a neural network structured as classifier
    ----------------------------------------------------------------------------------------------------- """


    def _get_init( self ):
        """ -------------------------------------------------------------------------------------------------
        Return an Initializer according to the object attribute

        return:         [keras.initializers.Initializer]
        ------------------------------------------------------------------------------------------------- """
        if self.k_initializer == 'RUNIF':
            return initializers.RandomUniform( minval=-0.05, maxval=0.05, seed=cnfg[ 'seed' ] )
        
        if self.k_initializer == 'GLOROT':
            return initializers.glorot_normal( seed=cnfg[ 'seed' ] )

        if self.k_initializer == 'HE':
            return initializers.he_normal( seed=cnfg[ 'seed' ] )

        ms.print_err( "Initializer {} not valid".format( self.k_initializer ) )



    def _get_regul( self ):
        """ -------------------------------------------------------------------------------------------------
        Return a Regularizer according to the object attribute

        return:         [keras.regularizers.Regularizer]
        ------------------------------------------------------------------------------------------------- """
        if self.k_regularizer == 'L2':
            return regularizers.l2( 0 )     # FIXME REGUL WITH FACTOR=0 MAKES NO SENSE!

        if self.k_regularizer == 'NONE':
            return None

        ms.print_err( "Regularizer {} not valid".format( self.k_regularizer ) )



    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model.

        When using multi_gpu_model, Keras recommends to instantiate the model under a CPU device scope,
        so that the model's weights are hosted on CPU memory.
        Otherwise they may end up hosted on a GPU, which would complicate weight sharing.

        https://keras.io/utils/#multi_gpu_model
        ------------------------------------------------------------------------------------------------- """
        self.arch_layout        = None          # [str] code describing the order of layers in the model

        self.img_size           = None          # [list] use 'channels_last' convention
        self.ref_model          = None          # [str] optional reference model from which to load weights
        self.k_initializer      = None          # [str] code specifying the type of convolution initializer
        self.k_regularizer      = None          # [str] code specifying the type of convolution regularizer

        self.conv_filters       = None          # [list of int] number of kernels for each convolution
        self.conv_kernel_size   = None          # [list of int] (square) size of kernels for each convolution
        self.conv_strides       = None          # [list of int] stride for each convolution
        self.conv_padding       = None          # [list of str] padding (same/valid) for each convolution
        self.conv_activation    = None          # [list of str] activation function for each convolution
        self.conv_train         = None          # [list of bool] False to lock training of each convolution

        self.pool_size          = None          # [list of int] pooling size for each MaxPooling

        self.dnse_size          = None          # [list of int] size of each dense layer
        self.dnse_activation    = None          # [list of str] activation function for each dense layer
        self.dnse_train         = None          # [list of bool] False to lock training of each dense layer


        # initialize class attributes with values from cnfg dict
        for k in self.__dict__:
            if k not in cnfg:
                ms.print_err( "Attribute '{}' of class '{}' not indicated".format( k, self.__class__ ) )
            exec( "self.{} = cnfg[ '{}' ]".format( k, k ) )                              

        # check if the string defining the architecture layout contains incorrect chars
        s0      = set( layer_code.values() )    # accepted chars
        s1      = set( self.arch_layout )       # chars passed as config
        if s1 - s0:
            ms.print_err( "Incorrect code {} for architecture layout".format( self.arch_layout ) )

        # check if the architecture layout is well defined
        if self.arch_layout.count( layer_code[ 'flat' ] ) > 1:
            ms.print_wrn( "Multiple flatten layer found. The architecture may be ill-defined" )
        if self.arch_layout.count( layer_code[ 'rshp' ] ) > 1:
            ms.print_wrn( "Multiple reshape layer found. The architecture may be ill-defined" )
 
        # keep a global count of layers per kind, to ensure different names for layers that are at
        # the same level in the architecture, but in different branches
        self.i_conv             = 1
        self.i_pool             = 1
        self.i_dnse             = 1
        self.i_dcnv             = 1

        # create model
        if cnfg[ 'n_gpus' ] > 1:
            with tf.device( '/cpu:0' ):
                self.model  = self.define_model( mname=cnfg[ 'data_class' ] )
        else:
            self.model  = self.define_model( mname=cnfg[ 'data_class' ] )

        assert cnfg[ 'n_conv' ] == len( self.conv_filters ) == len( self.conv_kernel_size ) == \
                len( self.conv_strides ) == len( self.conv_padding ) == len( self.conv_activation ) == \
                len( self.conv_train )

        assert cnfg[ 'n_dnse' ] == len( self.dnse_size ) == len( self.dnse_activation ) == len( self.dnse_train )
        assert cnfg[ 'n_pool' ] == len( self.pool_size )

        # load from a possible reference model
        if self.ref_model is not None:
            if not load_model( self.model, self.ref_model ):
                ms.print_err( "Failed to load weights from {}".format( self.ref_model ) )

        model_summary( self.model, fname=cnfg[ 'data_class' ] )


    def define_model( self, mname='Classifier' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the classifier

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        i_conv, i_pool, i_dnse  = 3 * [ 0 ]                         # for keeping count
        enc_arch_layout         = self.arch_layout.split( layer_code[ 'stop' ] )[ 0 ]

        init        = self._get_init() 
        kreg        = self._get_regul()

        # INPUT LAYER
        self.input          = layers.Input( shape=self.img_size )   # height, width, channels
        x                   = self.input

        for i, layer in enumerate( enc_arch_layout ):

            # CONVOLUTIONAL LAYER
            if layer == layer_code[ 'conv' ]:
                x       = layers.Conv2D(
                    self.conv_filters[ i_conv ],                            # number of filters
                    kernel_size         = self.conv_kernel_size[ i_conv ],  # size of window
                    strides             = self.conv_strides[ i_conv ],      # stride (window shift)
                    padding             = self.conv_padding[ i_conv ],      # zero-padding around the image
                    activation          = self.conv_activation[ i_conv ],   # activation function
                    kernel_initializer  = init,
                    kernel_regularizer  = kreg,                             # TODO check also activity_regularizer
                    use_bias            = True,                             # TODO watch out for the biases!
                    trainable           = self.conv_train[ i_conv ],
                    name                = 'conv{}'.format( self.i_conv )
                )( x )
                i_conv      += 1
                self.i_conv += 1

            # MAX POOLING LAYER
            elif layer == layer_code[ 'pool' ]:
                x       = layers.MaxPooling2D(                          
                    pool_size       = self.pool_size[ i_pool ],             # pooling size
                    padding         = self.conv_padding[ i_pool ],          # zero-padding around the image
                    name            = 'pool{}'.format( self.i_pool )
                )( x )
                i_pool      += 1
                self.i_pool += 1

            # DENSE LAYERs
            elif layer == layer_code[ 'dnse' ]:
                x           = layers.Dense(                             
                    self.dnse_size[ i_dnse ],                              # dimensionality of the output
                    activation      = self.dnse_activation[ i_dnse ],       # activation function
                    trainable       = self.dnse_train[ i_dnse ],
                    name            = 'dnse{}'.format( self.i_dnse )
                )( x )
                i_dnse      += 1
                self.i_dnse += 1

            # FLATTEN LAYER
            # NOTE it supposes a single flatten layer in the architecture
            elif layer == layer_code[ 'flat' ]:
                x       = layers.Flatten( name='flat' )( x )

            else:
                ms.print_err( "Layer code '{}' not valid".format( layer ) )

        c               = layers.Dense( 2, activation='softmax', name='class' )( x )

        return models.Model(
                    inputs      = self.input,
                    outputs     = c,
                    name        = mname
        )

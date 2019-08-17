"""
#############################################################################################################

process objects search in LAVA images using keras model, and store the probabilities
in .h5 files

    alex    2019

#############################################################################################################
"""

import  os
import  sys
import  numpy
import  h5py

from keras.models       import load_model

import  lava_geo        as lg
import  kspa_model

verbose     = 1                                         # verbose level
image_size  = ( 720, 540 )                              # size of LAVA images
models      = {}                                        # dictionary with Keras models
mod_dir     = "../keras/model_dir"                      # directory with Keras models
pro_dir     = "prob"                                    # directory where to write probabilities
mname       = "nn_best.h5"                              # standard name of model file
sentences   = "lists/with/selected_sentences_with.txt"  # file with list of sentences


def setup():
    """
    set up the external information in lava_geo, read the Keras models
    """
    global models

    lg.setup()
    lg.verbose  = verbose
    for c in lg.categories:
        models[ c ] = load_model( os.path.join( mod_dir, c, mname ) )
    if not os.path.exists( pro_dir ):
        os.makedirs( pro_dir )


def recall_batch( images, category ):
    """
    recall the network on a batch of images, returning probabilities of the object
    to be present in the images on the horizontal axis
    """
    images  = images[ category ]
    nn      = models[ category ]
    if not isinstance( images, numpy.ndarray ):
        if verbose:
            print( "Error in recall_batch: invalid format for images" )
        return None
    if len( images.shape ) != 4:    
        if verbose:
            print( "Error in recall_batch: invalid format for images" )
        return None

    r   = nn.predict( images )
    r   = r[ :, 0 ]
    r   = numpy.reshape( r, lg.n_coords[ category ] )
    r   = r.max( axis=1 )
    return r


def search_objs( fname, categories ):
    """
    search for objects of several given categories in the image file fname
    if categories is None uses all categories
    """
    prob    = {}
    scans   = lg.read_scans( fname )
    for c in categories:
        prob[ c ]        = recall_batch( scans, c )
    return prob


def save_h5( img, prob ):
    """
    save probabilities on h5 file
    """
    fname   = os.path.join( pro_dir, img + '.h5' )
    f       = h5py.File( fname, 'w' )
    for c in prob.keys():
        f.create_dataset( c, data=prob[ c ] )
    f.close()


def prob_sentence( sentence ):
    """
    given a LAVA sentence, extract probabilites of objects by running Keras models
    """
    img, comp, cats = kspa_model.read_sentence( sentence )
    amb             = kspa_model.ambigue( comp, cats )
    if comp is None:
        return
    prob    = search_objs( img, ( comp, amb[ 0 ], amb[ 1 ] ) )
    save_h5( img, prob )
    if verbose:
        print( img )


def prob_sentences( sentences ):
    """
    apply show_searches to all images in the corpus
    """
    if not os.path.isfile( sentences ):
        print( "Error: file {} not found".format( sentences ) )
        sys.exit()
    with open( sentences ) as fin:
        for sentence in fin:
            prob_sentence( sentence )


def main( argv ):
    """
    if there is an argument, it is assumed to be the name of a file with sentences
    """
    setup()
    if len( argv ) > 1:
        prob_sentences( argv[ 1 ] )
    else:
        prob_sentences( sentences )


"""
if the file is executed, run the main
"""

if __name__ == '__main__':
    main( sys.argv )

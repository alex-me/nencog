"""
#############################################################################################################

display results of objects search in LAVA images
using keras model

    alex    2019

#############################################################################################################
"""

import  os
import  sys
import  numpy

from keras.models   import load_model
from matplotlib     import pyplot
from matplotlib     import image

import  lava_geo    as lg

verbose     = 1                                     # verbose level
image_size  = ( 720, 540 )                          # size of LAVA images
models      = {}                                    # dictionary with Keras models
mod_dir     = "../keras/model_dir"                  # directory with Keras models
img_dir     = "images"                              # directory where to write images
mname       = "nn_best.h5"                          # standard name of model file
plot_scale  = 200                                   # scaling factor for probability plots
colors      = {                                     # colors for plots
    'person'    : '#3cb371',
    'chair'     : '#ff69b4',
    'bag'       : '#dc143c',
    'telescope' : '#8a2be2'
}


def setup():
    """
    set up the external information in lava_geo, read the Keras models
    """
    global models

    lg.setup()
    lg.verbose  = verbose
    for c in lg.categories:
        models[ c ] = load_model( os.path.join( mod_dir, c, mname ) )
    if not os.path.exists( img_dir ):
        os.makedirs( img_dir )


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


def scale_plot( y ):
    """
    scale the plots of detection probability so to display well
    under the image
    """
    y0      = 1.1 * plot_scale + image_size[ 1 ]
    return  y0 - plot_scale * y


def fnames( fname ):
    """
    return full path names for input and output images
    """
    fin     = os.path.join( lg.ldir, lg.idir, fname + lg.ext )
    fout    = os.path.join( img_dir, fname + lg.ext )
    return fin, fout


def search_objs( fname ):
    """
    search for objects of several given categories in the image file fname
    if categories is None uses all categories
    """
    obj     = {}
    scans   = lg.read_scans( fname )
    for c in lg.categories:
        obj[ c ]        = recall_batch( scans, c )
    return obj


def show_searches( fname ):
    """
    show results of a search of multiple categories by overlaying boxes on the image
    """
    x           = lg.x_coordinates
    fin, fout   = fnames( fname )
    img         = image.imread( fin )
    objs        = search_objs( fname )
    fig, ax = pyplot.subplots()
    ax.imshow( img )
    for c in objs.keys():
        y   = scale_plot( objs[ c ] )
        pyplot.plot( x, y, color=colors[ c ] )
    pyplot.axis( 'off' )
    pyplot.savefig( fout, bbox_inches='tight', dpi=200, pad_inches=0.01 )
    pyplot.close()
    if verbose:
        print( "written image " + fname )


def show_all_searches():
    """
    apply show_searches to all images in the corpus
    """
    files   = os.listdir( os.path.join( lg.ldir, lg.idir ) )
    for f in files:
        show_searches( os.path.splitext( f )[ 0 ] )


def main( argv ):
    """
    if there is an argument, it is assumed to be the name of an image to process
    if the argument is 'all' then all images are processed
    """
    setup()
    if len( argv ) > 1:
        if argv[ 1 ] == "all" or argv[ 1 ] == "ALL":
            show_all_searches()
        else:
            show_searches( argv[ 1 ] )


"""
if the file is executed, run the main
"""

if __name__ == '__main__':
    main( sys.argv )

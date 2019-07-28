"""
#############################################################################################################

Search for objects in LAVA images
objects should belong to fixed categories, for each one there is a box size, and a trained
neural model that recognize if the object is inside the object or not

    alex    2019 version for Keras models - adapted from ../search_lava.py

#############################################################################################################
"""

import  os
import  sys
import  numpy

import  pack_lava
import  put_boxes

from PIL            import Image
from keras.models   import load_model

verbose = 1                             # verbose level
mod_dir = "model_dir"                   # directory with models
mname   = "nn_best.h5"                  # standard name of model file
models  = {}                            # dictionary with Keras models
ldir    = "../../lava/corpus"           # main directory of original LAVA
idir    = "images"                      # subdirectory with images
ext     = ".png"                        # image filenames extension
limits  = ( 30, 30, 690, 510 )          # limits of the useful region of images (x0, y0, x1, y1)
step    = 20                            # step of image scanning
n_hits  = 10                            # number of accepted hits in object search
threshs = {                             # minimum probability for accepting a hit
    'person' :      0.98,
    'chair' :       0.98,
    'bag' :         0.97,
    'telescope' :   0.99
}


def set_pb():
    """
    relativize paths in put_boxes to the current directory
    """
    put_boxes.ldir      = ldir
    put_boxes.annfile   = '../' + put_boxes.annfile


def setup( categories=None ):
    """
    perform a general setup:
        - relativize paths in put_boxes to the current directory
        - read all nn models

    categories:     [list] list of categories for which models should be built, if None use pack_lava list
    """
    global models

    set_pb()

    if categories is None:
        categories  = pack_lava.categories
    for c in categories:
        models[ c ] = load_model( os.path.join( mod_dir, c, mname ) )


def read_image( fname ):
    """
    read one image, using as fname the visualFilename field in lava.json
    return a numpy array
    """
    fimg    = os.path.join( ldir, idir, fname + ext )
    return pack_lava.read_image( fimg, graylevel=False )


def scan_image( img, category='person' ):
    """
    get samples displacing a box around the given one, and saving the samples as image files
    the size of the samples is forced to be standard, centered horizontally with respect to
    the given box (but not vertically)
    """
    x0, y0, x1, y1  = limits
    w, h            = pack_lava.sizes[ category ]

    win             = []
    coords          = []
    top             = y0
    bottom          = top + h
    while bottom < y1:
        left        = x0
        right       = left + w
        while right < x1:
            win.append( img[ top:bottom, left:right, : ] )
            coords.append( ( left, top ) )
            left        += step
            right       = left + w
        top         += step
        bottom          = top + h

    return { "windows" : numpy.array( win ), "coordinates" : coords }



def recall_batch( images, nn ):
    """
    recall the network on a batch of images, returning probabilities of the object
    to be present in the images
    """
    if not isinstance( images, numpy.ndarray ):
        if verbose:
            print( "Error in recall_batch: invalid format for images" )
        return None
    if len( images.shape ) != 4:    
        if verbose:
            print( "Error in recall_batch: invalid format for images" )
        return None

    r   = nn.predict( images )
    """
    print( "r :" )
    print( r )
    """
    nr  = 1 + ( r - r.max() ) / r.ptp()
    return nr[ :, 0 ]


def obj_prop( coords, scores, n_hits=n_hits, thresh=0.8 ):
    """
    return a shortlist of boxes proposals that may contain objects
    """
    idx     = numpy.flipud( scores.argsort() )
    rsort   = scores[ idx[ : n_hits ] ]
    csort   = coords[ idx[ : n_hits ] ]
    if verbose > 1:
        print( "confidence over {} hits is in range {:4.2f}-{:4.2f}".format( n_hits, rsort[ 0 ], rsort[ -1 ] ) )
    n       = ( rsort > thresh ).sum()
    return csort[ : n ], rsort[ : n ]


def search_objs( img, categories=None, n_hits=n_hits ):
    """
    search for objects of several given categories in the image img
    if categories is None uses all categories
    """
    obj     = {}
    if categories is None:
        categories      = pack_lava.categories
    for category in categories:
        scans           = scan_image( img, category=category )
        wins            = scans[ "windows" ]
        coords          = numpy.array( scans[ "coordinates" ] )
        res             = recall_batch( wins, models[ category ] )
        obj[ category ] = obj_prop( coords, res, thresh=threshs[ category ] )
    return obj


def append_boxes( coords, category='person', box=[], obj=[] ):
    """
    append the coordinates coords to the list of boxes
    do not return, just append in place
    """
    w, h    = pack_lava.sizes[ category ]
    for x0, y0 in coords:
        x1  = x0 + w
        y1  = y0 + h
        box.append( ( x0, y0, x1, y1 ) )
        obj.append( category )


def show_searches( fname, categories=None, n_hits=n_hits ):
    """
    show results of a search of multiple categories by overlaying boxes on the image
    """
    img     = read_image( fname )
    objs    = search_objs( img, categories=categories, n_hits=n_hits )
    box     = []
    obj     = []
    for category in objs.keys():
        append_boxes( objs[ category ][ 0 ], category=category, box=box, obj=obj )
    fname   = put_boxes.annotate_image( ( fname, obj, box ) )
    if verbose:
        print( "written annotated image " + fname )


def show_all_searches():
    """
    apply show_searches to all images that have been manually annotated
    """
    anns    = put_boxes.read_annotations()
    for ann in anns:
        show_searches( ann[ 0 ] )


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

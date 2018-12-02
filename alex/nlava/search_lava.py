"""
search for objects in LAVA images
objects should belong to fixed categories, for each one there is a box size, and a trained
neural model that recognize if the object is inside the object or not

alex    Nov  2018
"""

import  os
import  sys
import  numpy

import  cnn_lava
import  pack_lava
import  put_boxes

from PIL    import Image

verbose = 1                         # verbose level
ldir    = "../lava/corpus/images"   # main directory of original LAVA
ext     = ".png"                    # image filenames extension
limits  = ( 30, 30, 690, 510 )      # limits of the useful region of images (x0, y0, x1, y1)
step    = 20                        # step of image scanning
n_hits  = 10                        # number of accepted hits in object search
threshs = {                         # minimum probability for accepting a hit
    'person' :      0.90,
    'chair' :       0.65,
    'bag' :         0.80,
    'telescope' :   0.80
}


def setup():
    """
    perform a general setup, including reading all nn models
    """
    cnn_lava.setup()
    cnn_lava.restore_all_nn()
    cnn_lava.verbose    = verbose


def read_image( fname ):
    """
    read one image, using as fname the visualFilename field in lava.json
    return a numpy array
    """
    fimg    = os.path.join( ldir, fname + ext )
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


def obj_prop( coords, scores, n_hits=n_hits, thresh=0.8 ):
    """
    return a shortlist of boxes proposals that may contain objects
    """
    idx     = numpy.flipud( scores.argsort() )
    rsort   = scores[ idx[ : n_hits ] ]
    csort   = coords[ idx[ : n_hits ] ]
    if verbose > 1:
        print( "confidence over {} hits is in range {:4.2f}-{:4.2f}".format( n_hits, rsort[ 0 ], rsort[ -1 ] ) )
    n       = numpy.argmin( rsort > thresh )
    return csort[ : n ], rsort[ : n ]


def search_obj( img, category='person', n_hits=n_hits, thresh=0.8 ):
    """
    search for objects of the given category in the image img
    """
    scans   = scan_image( img, category=category )
    wins    = scans[ "windows" ]
    coords  = numpy.array( scans[ "coordinates" ] )
    res     = cnn_lava.recall_batch( wins )
    return obj_prop( coords, res )


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
        res             = cnn_lava.recall_cat_batch( wins, category=category )
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


def show_search( fname, category='person', n_hits=n_hits, thresh=0.8 ):
    """
    show results of a search by overlaying boxes on the image
    """
    img     = read_image( fname )
    c, r    = search_obj( img, category=category, n_hits=n_hits, thresh=thresh )
    box     = []
    obj     = []
    append_boxes( c, category=category, box=box, obj=obj )
    fname   = put_boxes.annotate_image( ( fname, obj, box ) )
    if verbose:
        print( "written annotated image " + fname )


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
    if there is an argument, take it as image filename, otherwise ask for an image file
    on this filename performs object search for all categories
    """
    setup()
    if len( argv ) >1:
        show_searches( argv[ 1 ] )
    else:
        f   = input( "enter image filename [<return> to exit]: " )
        while len( f ):
            show_searches( f )
            f   = input( "enter image filename [<return> to exit]: " )


"""
if the file is executed in non interactive mode, run the main

if __name__ == '__main__':
    main( sys.argv )
"""

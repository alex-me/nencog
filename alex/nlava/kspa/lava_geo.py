"""
#############################################################################################################

set several geometrical attributes useful for searching objects in LAVA images
objects should belong to fixed categories, for each one there is a specific box size
the scansion of the images are performed with a fixed number of steps (and setp size)
in the horizontal dimension, a requirement useful for handling search results in the Nengo model
in the vertical dimension the setp size is fixed, but the number of steps varies according to
the object category

    alex    2019

#############################################################################################################
"""

import  os
import  sys
import  numpy

from PIL            import Image

verbose     = 1                             # verbose level
ldir        = "../../lava/corpus"           # main directory of original LAVA
idir        = "images"                      # subdirectory with images
ext         = ".png"                        # image filenames extension
limits      = ( 69, 30, 655, 510 )          # limits of the useful region of images (x0, y0, x1, y1)
step        = 20                            # step of image scanning
categories  = ( 'person', 'chair', 'bag', 'telescope' )
sizes       = {                         # size of the standard samples for each category
    'person' :      ( 128, 360 ),
    'chair' :       ( 128, 200 ),
    'bag' :         (  80, 140 ),
    'telescope' :   ( 100,  64 )
}
coordinates = {                             # the coordinates of the subwindows for each category
    'person' :      None,
    'chair' :       None,
    'bag' :         None,
    'telescope' :   None
}
n_coords    = {
    'person' :      None,
    'chair' :       None,
    'bag' :         None,
    'telescope' :   None
}
shapes      = {
    'person' :      None,
    'chair' :       None,
    'bag' :         None,
    'telescope' :   None
}

x_coordinates   =   []                      # the coordinates common to all categories, along X


def set_coords( category='person' ):
    """
    generate the coordinates of the subwindows to be extracted from an image
    the cordinates refers to the center of the box horizontally, but not vertically
    returns also the number of coordinates for the two dimensions
    """
    x0, y0, x1, y1  = limits
    w, h            = sizes[ category ]
    coords          = []

    n_xcoords   = ( x1 - x0 ) // step
    for i in range( n_xcoords ):
        middle      = x0 + i * step
        top         = y0
        bottom      = top + h
        while bottom < y1:
            coords.append( ( middle, top ) )
            top         += step
            bottom      = top + h
    n_ycoords       = len( coords ) // n_xcoords

    return coords, ( n_xcoords, n_ycoords )


def setup():
    """
    perform a general setup, executing set_coords() for all cetagories
    """
    global x_coordinates

    for c in categories:
        xy, nxy             = set_coords( c )
        n                   = nxy[ 0 ] * nxy[ 1 ]
        coordinates[ c ]    = xy
        n_coords[ c ]       = nxy
        shapes[ c ]         = ( n, sizes[ c ][ 1 ], sizes[ c ][ 0 ], 3 )

    x               = list( set( [ c[ 0 ] for c in coordinates[ 'person' ] ] ) )
    x.sort()
    x_coordinates   = x



def scan_image( img, category='person' ):
    """
    get samples displacing a box around the given one, and saving the samples as image files
    the size of the samples is forced to be standard, centered horizontally with respect to
    the given box (but not vertically)
    """
    w, h            = sizes[ category ]
    win     = []
    for x, y in coordinates[ category ]:
        top         = y
        bottom      = top + h
        left        = x - w // 2
        right       = left + w
        win.append( img[ top:bottom, left:right, : ] )

    return numpy.array( win )


def read_image( fname ):
    """
    read one image, using as fname the visualFilename field in lava.json
    return a numpy array
    """
    f       = os.path.join( ldir, idir, fname + ext )
    if not os.path.exists( f ):
        if verbose:
            print( "Error in read_image: image " + f + " not found" )
        return None
    img     = Image.open( f )
    img     = numpy.asarray( img, dtype=numpy.float32 )
    img     = numpy.multiply( img, 1.0 / 255.0 )
    return img


def read_scans( fname ):
    """
    read one LAVA image and generate batches of subwindows, specific for each category

    return a dictionary with batches and coordinates for all categories
    """
    inputs  = {}
    img     = read_image( fname )
    for c in categories:
        inputs[ c ] = scan_image( img, c )
        
    return inputs


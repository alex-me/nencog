"""
extract samples from the original images, containing one single object each
images are cropped using annotations of object category and coordinates, with
several replications around each original box. The number of variations are such as to
keep the overall number of samples per category similar.
Samples have a standard size (different for categories), derived by the statistics extracted
from the annotated coordinates with coord_stat.py

alex    Sep  2018
"""

import  os
import  sys

from PIL    import Image

ldir    = "../lava/corpus/images"   # main directory of original LAVA
sdir    = "samples"                 # local directory of output samples
annfile = "coords/coords.txt"       # file with image annotations
ext     = ".png"                    # image filenames extension

categories  = ( 'person', 'chair', 'bag', 'telescope' )
persons     = ( 'Andrei', 'Danny', 'Yevgeni' )
sizes       = {                     # size of the standard samples for each category
    'person' :      ( 128, 360 ),
    'chair' :       ( 128, 200 ),
    'bag' :         (  80, 140 ),
    'telescope' :   ( 100,  64 )
}
var_span    = {                     # range of variations around an original box when cropping samples
    'person' :      range( -1, 2 ),
    'chair' :       range( -2, 3 ),
    'bag' :         range( -2, 3 ),
    'telescope' :   range( -3, 4 )
}

file_counts  = {                    # progressive numbering of sample files for each category
    'person' :      0,
    'chair' :       0,
    'bag' :         0,
    'telescope' :   0
}

file_fmt    = "f_{:05d}.png"        # formatting string for generating the proper filename of samples

def check_dirs():
    """
    verify that directories for samples are in place, otherwise create them
    """
    for c in categories:
        d   = os.path.join( sdir, c )
        if not os.path.exists( d ):
            os.makedirs( d )


def category( name ):
    """
    return a valid category for the object named by name
    """
    if name in persons:
        return 'person'
    return name


def read_annotation( line ):
    """
    read one line with annotated objects and coordinates of an image
    return a tuple made by:
        image name, list with object classes, list with coordinates
    """
    objs    = []
    coords  = []
    fields  = line.split()
    fimg    = fields[ 0 ]
    fields  = fields[ 1 : ]
    if len( fields ) % 5:
        print( "ERORR in read_annotation, invalid line content:", line )
        sys.exit( 1 )
    if len( fields ) < 5 * 3:
        print( "WARNING in read_annotation, less than 3 objects in image:", fimg )
    n_obj   = len( fields ) // 5
    for o in range( n_obj ):
        i           = o * 5
        objs.append( category( fields[ i ] ) )
        coords.append( tuple( [ int( f ) for f in fields[ i + 1 : i + 5 ] ] ) )
        
    return ( fimg, objs, coords )


def read_annotations( fname=None ):
    """
    read the file with annotated objects and coordinates for every image
    """
    if fname is None:
        fname   = annfile
    with open( fname ) as annotations:
        im_boxes    = [ read_annotation( l ) for l in annotations ]
    return im_boxes


def in_image( img, box ):
    """
    check if the given box is contained by the image
    """
    x0, y0, x1, y1  = box
    width, height   = img.size
    if x0 < 0:          return False
    if y0 < 0:          return False
    if x1 >= width:     return False
    if y1 >= height:    return False
    return True


def get_sample( img, obj, box ):
    """
    get samples displacing a box around the given one, and saving the samples as image files
    the size of the samples is forced to be standard, centered horizontally with respect to
    the given box (but not vertically)
    """
    x0, y0, x1, y1  = box
    var             = var_span[ obj ]
    width, height   = sizes[ obj ]
    count           = file_counts[ obj ]
    out_dir         = os.path.join( sdir, obj )
    x_c             = ( x0 + x1 ) / 2.
    x0              = int( x_c - width / 2. )

    for x in var:
        s_x0    = x0 + x
        s_x1    = s_x0 + width
        for y in var:
            s_y0    = y0 + y
            s_y1    = s_y0 + height
            crop    = s_x0, s_y0, s_x1, s_y1
            if not in_image( img, crop ):
                continue
            count   += 1
            fname   = os.path.join( out_dir, file_fmt.format( count ) )
            sample  = img.crop( crop )
            sample.save( fname )

    file_counts[ obj ]  = count


def get_samples_image( annotation ):
    """
    get samples from one image with boxes of objects
    """
    fimg, obj, box  = annotation
    f_in            = os.path.join( ldir, fimg + ext )
    img             = Image.open( f_in )
    for o, b in zip( obj, box ):
        get_sample( img, o, b )


def get_all_samples( annotations ):
    """
    get samples from all images in the list annotations (in the format returned by read_annotations() )
    """
    check_dirs()
    for a in annotations:
        get_samples_image( a )


def main( argv ):
    """
    do all
    """
    a   = read_annotations()
    get_all_samples( a )


"""
if the file is executed in non interactive mode, run the main
"""
if __name__ == '__main__':
    main( sys.argv )

"""
extract samples from the original images, to be used as no-object by the neural models
images are cropped using list of images for each category and range of horizontal coordinates
void of objects of the corresponding category
Samples have the same standard size (different for categories) as in extract_samples.py

alex    Sep  2018
"""

import  os
import  sys
import  random

from PIL    import Image

ldir    = "../lava/corpus/images"   # main directory of original LAVA
sdir    = "samples"                 # local directory of output samples
annfile = "coords/no_samples.txt"   # file with image lists and coordinate range
ext     = ".png"                    # image filenames extension

categories  = ( 'person', 'chair', 'bag', 'telescope' )
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

file_fmt    = "n_{:05d}.png"        # formatting string for generating the proper filename of samples

y_limits    = ( 0, 540 )            # allowed vertical range (the entire image)

n_samples   = 625                   # number of samples in each image, this value is tuned for ~10000 samples


def read_annotation( line ):
    """
    read one line with annotated objects and coordinates of an image
    return a tuple made by:
        image name, object classes, tuple with coordinate range
    """
    fields  = line.split()
    if len( fields ) != 4:
        print( "ERORR in read_annotation, invalid line content:", line )
        sys.exit( 1 )
    fimg    = fields[ 0 ]
    obj     = fields[ 1 ]
    coords  = int( fields[ 2 ] ), int( fields[ 3 ] )
        
    return ( fimg, obj, coords )


def read_annotations( fname=None ):
    """
    read the file with annotated objects and coordinates for every image
    """
    if fname is None:
        fname   = annfile
    with open( fname ) as annotations:
        im_boxes    = [ read_annotation( l ) for l in annotations ]
    return im_boxes


def get_sample( img, obj, x_limits ):
    """
    get samples displacing a box in the allowed range, and saving the samples as image files
    bozes are place randomly avoiding duplicate boxes
    the size of the samples is forced to be standard
    """
    width, height   = sizes[ obj ]
    count           = file_counts[ obj ]
    out_dir         = os.path.join( sdir, obj )
    x_range         = range( x_limits[ 0 ], x_limits[ 1 ] - width )
    y_range         = range( y_limits[ 0 ], y_limits[ 1 ] - height )
    xy              = random.sample( [ ( x, y ) for x in x_range for y in y_range ], 2 * n_samples )
    u_xy            = set( xy )
    while len( u_xy ) < n_samples:
        u_xy    |= set( random.sample( [ ( x, y ) for x in x_range for y in y_range ], 2 * n_samples ) )
    xy  =list( u_xy )[ : n_samples ]

    for x, y in xy:
        x1      = x + width
        y1      = y + height
        crop    = x, y, x1, y1
        count   += 1
        fname   = os.path.join( out_dir, file_fmt.format( count ) )
        print( fname + '\t', crop )
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
    get_sample( img, obj, box )


def get_all_samples( annotations ):
    """
    get samples from all images in the list annotations (in the format returned by read_annotations() )
    """
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

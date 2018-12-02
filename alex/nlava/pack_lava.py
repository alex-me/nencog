"""
pack the LAVA samples, as extracted by extract_samples.py and extract_no_samples.py
into datasets, formatted as numpy tensors
note that less samples than those available can be packed into datasets, due to both
limitations in the pickle protocols, and in the critical usage of RAM.

in the current version, max_samples = 2**11 works with python 3.5 and 16GB RAM

each category has its own dataset, with boolean labels for object / no-object

alex    Nov 2018
"""


import  pickle
import  gzip
import  os
import  numpy

from PIL    import Image

"""
main globals
"""
verbose     = 1                             # verbose level
sample_dir  = "./samples"                   # directory with all samples
dest        = "./data"                      # directory where to write datasets
d_prefix    = "lava_"                       # common prefix to dataset filenames
divisor     = 2**6                          # divisor of the final number of samples in the dataset
max_samples = 2**11                         # maximum numer of positive/negative samples in the dataset
img_formats = ( "jpg", "png", "gif" )       # supported image formats
true_pre    = 'f_'                          # prefix of a true sample filename
false_pre   = 'n_'                          # prefix of a false sample filename
categories  = ( 'person', 'chair', 'bag', 'telescope' )
category    = 'person'              # the current category
sizes       = {                     # size of the standard samples for each category
    'person' :      ( 128, 360 ),
    'chair' :       ( 128, 200 ),
    'bag' :         (  80, 140 ),
    'telescope' :   ( 100,  64 )
}



def _true_sample( fname ):
    """
    check if the sample is a "true" case, of a "false" one
    """
    if true_pre in fname:
        return True
    if false_pre in fname:
        return False
    if verbose:
            print( "Error: sample " + fname + " is not a true nor a false case" )
    return None



def read_image( f, graylevel=False ):
    """
    read one image with filename f, optionally force graylevel format,
    and return a numpy array that follows TF convention: ( height, width, channels )
    """
    if not os.path.exists( f ):
        if verbose:
            print( "Error in read_image: image " + f + " not found" )
        return None
    ext = f.split( '.' )[ -1 ]
    if not ext in img_formats:
        if verbose:
            print( "Error in read_image: image " + f + " not supported" )
        return None

    img     = Image.open( f )
    if graylevel and img.mode != 'L':
        img     = img.convert( 'L' )

    img     = numpy.asarray( img, dtype=numpy.float32 )
    img     = numpy.multiply( img, 1.0 / 255.0 )

    if len( img.shape ) == 2:
        img     = img.reshape( ( img.shape[ 0 ], img.shape[ 1 ], 1 ) )

    return img



def read_images( category='person', graylevel=False ):
    """
    read the original images
    the array is forced to be a multiple of divisor, and to contain an equal number of
    positive and negative samples
    """
    if not category in categories:
        if verbose:
            print( "Error in read_images: category " + category + " is missing" )
        return None
    src         = os.path.join( sample_dir, category )
    if not os.path.isdir( src ):
        if verbose:
            print( "Error in read_images: source directory " + src + " is missing" )
        return None
    imgs_yes    = []
    imgs_no     = []
    labs_yes    = []
    labs_no     = []
    yes         = numpy.array( ( 1.0, 0.0 ), dtype=numpy.float32 )
    no          = numpy.array( ( 0.0, 1.0 ), dtype=numpy.float32 )
    n_yes       = 0
    n_no        = 0
    nomore_y    = False
    nomore_n    = False
    
    files       = os.listdir( src )
    files.sort()
    for f in files:
        if nomore_y and nomore_n: continue
        i       = read_image( os.path.join( src, f ), graylevel=graylevel )
        if _true_sample( f ):
            if nomore_y: continue
            imgs_yes.append( i )
            labs_yes.append( yes )
            n_yes   += 1
            if n_yes > max_samples:
                nomore_y    = True
        else:
            if nomore_n: continue
            imgs_no.append( i )
            labs_no.append( no )
            n_no    += 1
            if n_no > max_samples:
                nomore_n    = True

    n       = min( n_yes, n_no )
    n       = divisor * ( n // divisor )
    imgs    = numpy.array( imgs_yes[ : n ] + imgs_no[ : n ] )
    labs    = numpy.array( labs_yes[ : n ] + labs_no[ : n ] )

    return { "images" : imgs, "labels" : labs }


def write_dataset( data, category='person' ):
    f       = d_prefix + category + '.gz'
    f       = os.path.join( dest, f )
    gf              = gzip.open( f, 'wb' )
    pickle.dump( data, gf, 3 )
    gf.close()


"""
# typical usage:
#
category    = 'telescope'
img_lab     = read_images( category=category )
write_dataset( img_lab, category=category )
"""

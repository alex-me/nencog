"""
overlay boxes over objects, according to the annotated file

alex    July 2018
        Sep  2018 adjusted for multiple 'person' category, and check for less than 3 object
"""

import  json
import  os
import  sys

from PIL    import Image, ImageDraw

ldir    = "../lava/corpus"      # main directory of original LAVA
idir    = "images"              # local directory of outptu images
annfile = "coords/coords.txt"   # file with image annotations
ext     = ".png"                # image filenames extension

colors  = {
    'person'    : 'MediumSeaGreen',
    'chair'     : 'HotPink',
    'bag'       : 'Crimson',
    'telescope' : 'BlueViolet'
}

persons = ( 'Andrei', 'Danny', 'Yevgeni' )


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


def draw_box( img, box, color, width=3 ):
    """
    draw one box
    """
    x0, y0, x1, y1  = box
    vert            = [
        ( x0, y0 ),
        ( x0, y1 ),
        ( x1, y1 ),
        ( x1, y0 ),
        ( x0, y0 )
    ]
    img.line( vert, width=width, fill=color )


def annotate_image( annotation ):
    """
    annotate one image with boxes of objects
    """
    fimg, obj, box  = annotation
    f_in            = os.path.join( os.path.join( ldir, idir ), fimg + ext )
    f_out           = os.path.join( idir, fimg + ext )
    img             = Image.open( f_in )
    img_box         = ImageDraw.Draw( img )
    for o, b in zip( obj, box ):
        draw_box( img_box, b, colors[ o ] )
    img.save( f_out )


def annotate_images( annotations ):
    """
    annotate all images in the list annotations (in the format returned by read_annotations() )
    """
    if not os.path.isdir( idir ):
        os.makedirs( idir )
    for a in annotations:
        annotate_image( a )



def main( argv ):
    """
    do all
    """
    a = read_annotations()
    annotate_images( a )

"""
if the file is executed in non interactive mode, run the main
"""
if __name__ == '__main__':
    main( sys.argv )

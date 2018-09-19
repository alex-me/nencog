"""
do some statistics over box coordinated, according to the annotated file

alex    Sep  2018
"""

import  os
import  sys
import  numpy

annfile     = "coords/coords.txt"   # file with image annotations
categories  = ( 'person', 'chair', 'bag', 'telescope' )
persons     = ( 'Andrei', 'Danny', 'Yevgeni' )


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
    fields  = fields[ 1 : ]
    n_obj   = len( fields ) // 5
    for o in range( n_obj ):
        i           = o * 5
        objs.append( category( fields[ i ] ) )
        coords.append( tuple( [ int( f ) for f in fields[ i + 1 : i + 5 ] ] ) )
        
    return ( objs, coords )


def read_annotations( fname=None ):
    """
    read the file with annotated objects and coordinates for every image
    """
    if fname is None:
        fname   = annfile
    boxes   = {}
    for c in categories:
        boxes[ c ]  = []
    with open( fname ) as annotations:
        for l in annotations:
            cat, box    = read_annotation( l )
            for c, b in zip ( cat, box ):
                boxes[ c ].append( b )
    for c in categories:
        boxes[ c ]  = numpy.array( boxes[ c ], dtype=float )

    return boxes


def stat_box( box ):
    """
    give statistics about boxes of one category
    """
    x0      = box[ :, 0 ]
    y0      = box[ :, 1 ]
    x1      = box[ :, 2 ]
    y1      = box[ :, 3 ]
    width   = x1 - x0
    height  = y1 - y0
    size    = width.mean() + width.std(), height.mean() + height.std()
    bounds  = x0.min(), y0.min(), x1.max(), y1.max()
    return { 'n_samples' : len( x0 ), 'size' : size, 'bounds' : bounds }


def pr_stats( boxes ):
    """
    print all statistics on coordinates
    """
    fmt = "{:10s}\t{:5d} samples\tsize {:5.1f}x{:5.1f}\tbounds {:3.0f}x{:3.0f} <-> {:3.0f}x{:3.0f}"
    for c in boxes.keys():
        stat    = stat_box( boxes[ c ] )
        print( fmt.format( c, stat[ 'n_samples' ], *stat[ 'size' ], *stat[ 'bounds' ] ) )


def main( argv ):
    """
    do all
    """
    boxes   = read_annotations()
    pr_stats( boxes )


"""
if the file is executed in non interactive mode, run the main
"""
if __name__ == '__main__':
    main( sys.argv )

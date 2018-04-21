"""
select a subset of categories from the CIFAR100 dataset
and generate a new dataset

the full CIFAR100 dataset can be downloaded with
curl -O http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

alex    May 2018

"""


import  pickle
import  gzip
import  os
import  sys
import  numpy
import  PIL.Image as Image


"""
main globals
"""
cifar_100   = "../../../tf/cifar10_data/cifar-100-python"           # original CIFAR100 directory
cifar_lbl   = "meta"                                                # original label names
cifar_trs   = "train"                                               # train set name
cifar_tss   = "test"                                                # test set name
cifar_sel   = "./cifar_data"                                        # output directory
cifar_png   = "./cifar_png"                                         # directory for sample images

classes     = (                                                     # categories for the selected dataset
    "apple",
    "bed",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bus",
    "butterfly",
    "can",
    "chair",
    "clock",
    "couch",
    "crab",
    "cup",
    "fox",
    "girl",
    "hamster",
    "house",
    "keyboard",
    "lamp",
    "man",
    "motorcycle",
    "mouse",
    "mushroom",
    "orange",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "road",
    "rose",
    "snail",
    "streetcar",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tractor",
    "train",
    "tulip",
    "turtle",
    "wardrobe",
    "woman"
)


def my_pickle( f ):
    """
    use the appropriate pickle arguments depending on Python version
    """
    if sys.version[ 0 ] == '2':
        return pickle.load( f )
    return pickle.load( f, encoding='latin1' )


def save_png( image, name ):
    """
    saves an image (represented as a numpy array) to PNG.
    """
    if not os.path.isdir( cifar_png ):
        os.mkdir( cifar_png )
    f_name      = os.path.join( cifar_png, name )
    img         = image.reshape( 3, 32, 32 )
    img         = img.transpose( 1, 2, 0 )
    image_pil   = Image.fromarray( numpy.uint8( img ) ).convert( 'RGB' )
    with open( f_name, 'w' ) as f:
        image_pil.save( f, 'PNG')


def read_data_sets():
    """
    read the original CIFAR-100 datasets and return a tuple with training set, test set, label names
    """
    f       = os.path.join( cifar_100, cifar_trs )
    fd      = open( f, 'rb' )
    tr_set  = my_pickle( fd )
    fd.close()
    f       = os.path.join( cifar_100, cifar_tss )
    fd      = open( f, 'rb' )
    ts_set  = my_pickle( fd )
    fd.close()
    f       = os.path.join( cifar_100, cifar_lbl )
    fd      = open( f, 'rb' )
    labels  = my_pickle( fd )
    fd.close()

    return tr_set, ts_set, labels[ 'fine_label_names' ]


def remap_label( l, labels ):
    """
    map the original numeric label into the new one
    input:
        l       old numeric label
        labels  original list of label names
    output:
        new numeric label
    """
    n   = labels[ l ]
    if not n in classes:
        raise Exception( "attempt to remap unexpected numeric label " + str( l  ) )
    return classes.index( n )


def select_data( data, labels ):
    """
    select data and label for the only categories listed in classes
    input:
        data    a CIFAR-100 dataset
        labels  the CIFAR-100 complete label names
    output:
        dictionary { 'data' : data, 'labels': labels }
    """
    l   = data[ 'fine_labels' ]
    d   = data[ 'data' ]
    idx = [ i for i, e in enumerate( l ) if labels[ e ] in classes ]
    nd  = d[ idx ]
    nl  = [ remap_label( l[ i ], labels ) for i in idx ]
    return { 'data' : nd, 'labels': nl }


def select_samples( dset, n=10 ):
    """
    select and save n sample images for each class
    input:
        dset    a selected dataset
    """
    d   = dset[ 'data' ]
    l   = dset[ 'labels' ]
    nc  = len( classes )
    cnt = n * numpy.ones( nc, dtype=int )

    for img, lab in zip( d, l ):
        if cnt[ lab ]:
            name        = "{:s}_{:02d}.png".format( classes[ lab ], cnt[ lab ] )
            save_png( img, name )
            cnt[ lab ]  -= 1
        if not cnt.sum():
            return True
    return False


def write_data_sets( train, test ):
    """
    write the personalized CIFAR-100 selection
    """
    if not os.path.isdir( cifar_sel ):
        os.mkdir( cifar_sel )
    f       = os.path.join( cifar_sel, cifar_trs + ".gz" )
    fd      = gzip.open( f, 'w' )
    pickle.dump( train, fd, 2 )
    fd.close()
    f       = os.path.join( cifar_sel, cifar_tss + ".gz" )
    fd      = gzip.open( f, 'w' )
    pickle.dump( test, fd, 2 )
    fd.close()


if __name__ == "__main__":
    """
    if called directly, execute the selection
    """
    old_tr, old_ts, lbs = read_data_sets()
    new_tr              = select_data( old_tr, lbs )
    new_ts              = select_data( old_ts, lbs )
    write_data_sets( new_tr, new_ts )
    select_samples( new_ts, 20 )

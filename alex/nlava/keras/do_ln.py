"""
#############################################################################################################

Utilities for handling the dataset image folders, by creating symbolic links

The Keras preprocessing.image.ImageDataGenerator methods require a specific folder structure,
That is why is more convinient to keep a single folder with all the actual frames, and then to generate
the specific folders with symbolic links of selected frames.

The structure should be something like:

    -------------------------------------------------------------------------------
    dataset_name/               main folder

        dset_class/             dataset for a single category of objects

            train/              training set
                yes/            images containing an object of the category
                no/             images that do not contain objects of the category
            valid/              validation set
                yes/            images containing an object of the category
                no/             images that do not contain objects of the category
    -------------------------------------------------------------------------------

    alex   2019

#############################################################################################################
"""

import  os
import  sys
import  shutil
import  random
import  numpy

DEBUG           = True
SEED            = 3

frac_train      = 0.80                                  # fraction of training set
frac_valid      = 0.20                                  # fraction of validation set
batch_msize     = 128                                   # NOTE max batch size that will be used during training

file_ext        = ( '.jpg', '.jpeg', '.png' )           # accepted image file formats
cond_ext        = lambda x: x.endswith( file_ext )      # condition selecting only image files

root_dir        = '../samples'
link_dir        = 'link'
categories      = ( 'person', 'chair', 'bag', 'telescope' )
classes         = ( 'yes', 'no' )                       # the two classes: there is an object / no object
cond_yes        = lambda x: 'f_' in x                   # condition selecting image files with objects
cond_no         = lambda x: 'n_' in x                   # condition selecting image files without objects
cond_class      = { 'yes' : cond_yes, 'no' : cond_no }  # conditions for selecting files into the two classes
warn            = "Directory {} does not exist"



def frac_dataset( f_train, f_valid, size, batch_size=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Return 2 slices dividing a dataset into training and validation sets.

    Each subset size is computed to be divisible by the 'batch_size' (or the largest one)
    that will be used during training. This because Keras functions prefer dataset like that...

    f_train:        [float] fraction of images to be used as training set
    f_valid:        [float] fraction of images to be used as validation set
    size:           [int] total number of images in the dataset
    batch_size:     [int] maximum size of the minibatch (if 1 consider no limitation to the size)

    return:         [list of slice]
    ----------------------------------------------------------------------------------------------------- """
    assert ( f_train + f_valid - 1 ) < 0.001

    f_tr        = int( batch_size * ( f_train * size // batch_size ) )
    f_vd        = int( batch_size * ( f_valid * size // batch_size ) )

    indx_train  = slice( 0,             f_tr )
    indx_valid  = slice( f_tr,          f_tr + f_vd )

    if DEBUG:
        print( "Total:\t{}\nTrain:\t{}\nValid:\t{}\n".format( size, f_tr, f_vd ) )
    
    return indx_train, indx_valid



def make_symlink( src_dir, dest_dir, files ):
    """ -----------------------------------------------------------------------------------------------------
    Make symbolic links of files from a folder to a second one

    NOTE: this assumes that, in case of multiple source folders, each of them has unique file names

    src_dir:        [str or list of str] source folder(s) containing original images
    dest_dir:       [str] destination folder
    files:          [list of str] name of files to be linked
    ----------------------------------------------------------------------------------------------------- """
    if isinstance( src_dir, str ):
        src_rel     = os.path.relpath( src_dir, dest_dir )          # get 'src' path relative to 'dest'
        for f in files:
            os.symlink( os.path.join( src_rel, f ), os.path.join( dest_dir, f ) )

    elif isinstance( src_dir, ( list, tuple ) ):
        src_rel     = [ os.path.relpath( sd, dest_dir ) for sd in src_dir ]
        for f in files:
            for i in range( len( src_dir ) ):
                if os.path.isfile( os.path.join( src_dir[ i ], f ) ):
                    os.symlink( os.path.join( src_rel[ i ], f ), os.path.join( dest_dir, f ) )



def dset_class( src_dir, dest_dir, size=None, batch_size=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Create a structured dataset made of simbolic links to image files.
    The dataset is for one category

    In case of multiple source folders, it assumes that files in each folder have unique names.

    src_dir:        [str] source folder containing original images
    dest_dir:       [str] destination folder
    size:           [int] amount of files to link (if None consider all files)
    batch_size:     [int] max allowed size of the minibatch (if 1 consider no limitation to the size)
    ----------------------------------------------------------------------------------------------------- """

    dest_train      = [ os.path.join( dest_dir, 'train/{}'.format( c ) ) for c in classes ]
    dest_valid      = [ os.path.join( dest_dir, 'valid/{}'.format( c ) ) for c in classes ]

    # remove the folder if already existed, and create a fresh one
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    for d in dest_train + dest_valid:
        os.makedirs( d )

    files       = {}
    for c in classes:
        files[ c ]  = []

    # get list of all files, for each class
    if not os.path.isdir( src_dir ):
        print( warn.format( src_dir ) )
        return

    for f in sorted( os.listdir( src_dir ) ):
        for c in classes:
            if cond_class[ c ]( f ) and cond_ext( f ):
                files[ c ].append( f )

    # randomly permute the file list, for each class
    random.seed( SEED )
    s_yes   = len( files[ 'yes' ] )
    s_no    = len( files[ 'no' ] )
    s       = min( s_yes, s_no ) if size is None else size
    size    = s if size is None else size
    indx    = random.sample( range( s ), size )
    for c in classes:
        files[ c ]  = numpy.array( files[ c ] )
        files[ c ]  = files[ c ][ indx ]

    # partition file list into the 3 subsets, for each class
    i_tr, i_vd      = frac_dataset( frac_train, frac_valid, s, batch_size=batch_size )
    files_train     = {}
    files_valid     = {}
    for c in classes:
        files_train[ c ]     = files[ c ][ i_tr ]
        files_valid[ c ]     = files[ c ][ i_vd ]

    # make symbolic links
    for i, c in enumerate( classes ):
        make_symlink( src_dir, dest_train[ i ], files_train[ c ] )
        make_symlink( src_dir, dest_valid[ i ], files_valid[ c ] )



if __name__ == '__main__':
    for c in categories:
        dset_class(
                os.path.join( root_dir, c ),
                os.path.join( link_dir, c ),
                batch_size  = batch_msize
        )

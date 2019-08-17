"""
#############################################################################################################

context model

disambiguate sentences of the LAVA dataset analyzing images, integrating Keras witgh Nengo SPA

this is the main module

    alex   2019

#############################################################################################################
"""

import  os
import  sys
import  argparse

import  kspa_model

def get_args():
    """ -----------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags
    
    return:         [dict] args (keys) and their values
    ----------------------------------------------------------------------------------------------------- """
    parser      = argparse.ArgumentParser()

    parser.add_argument(
            '-b',
            '--batch',
            action          = 'store_true',
            dest            = 'BATCH',
            help            = "Probabilities are read externally, from batch executions"
    )
    parser.add_argument(
            '-e',
            '--eval',
            action          = 'store',
            dest            = 'EVAL',
            type            = str,
            default         = None,
            help            = "Pathname of file with groud truths, should be used with --load"
    )
    parser.add_argument(
            '-f',
            '--full',
            action          = 'store',
            dest            = 'FULL',
            type            = str,
            default         = None,
            help            = "Pathname of file with full results, should be used with --eval"
    )
    parser.add_argument(
            '-l',
            '--load',
            action          = 'store',
            dest            = 'LOAD',
            type            = str,
            default         = None,
            help            = "Pathname of file with sentences to analyze"
    )
    parser.add_argument(
            '-m',
            '--method',
            action          = 'store',
            dest            = 'METHOD',
            type            = str,
            default         = "CLOSENESS",
            help            = "method to use for evaluating spatial closeness of objects"
    )
    parser.add_argument(
            '-n',
            '--ntsteps',
            action          = 'store',
            dest            = 'NTSTEPS',
            type            = int,
            default         = 50,
            help            = "number of timesteps in the Nengo simulation"
    )
    parser.add_argument(
            '-o',
            '--output',
            action          = 'store',
            dest            = 'OUTPUT',
            type            = str,
            default         = None,
            help            = "Pathname of file where results will be written"
    )
    parser.add_argument(
            '-p',
            '--plot',
            action          = 'store',
            dest            = 'PLOT',
            type            = float,
            default         = None,
            help            = "Plot SPA evolution for the specified time [seconds]"
    )
    parser.add_argument(
            '-s',
            '--separator',
            action          = 'store',
            dest            = 'SEPARATOR',
            type            = str,
            default         = ' <-> ',
            help            = "separator between the two words of the disambiguated output"
    )
    parser.add_argument(
            '-t',
            '--text',
            action          = 'store',
            dest            = 'TEXT',
            type            = str,
            default         = None,
            help            = "sentence to analyze (including image name)"
    )
    parser.add_argument(
            '-v',
            '--verbose',
            action          = 'count',
            dest            = 'VERBOSE',
            default         = 0,
            help            = "verbosity level, [-v] minumum messaging, [-vv] display Nengo progress bar"
    )

    return vars( parser.parse_args() )


def read_files():
    """ -----------------------------------------------------------------------------------------------------
    read files for evaluation mode
    
    return:         [tuple of lists] sentences and ground truths
    ----------------------------------------------------------------------------------------------------- """
    sentences   = []
    truths      = []
    with open( args[ 'LOAD' ] ) as fin:
        for line in fin:
            sentences.append( line )
    with open( args[ 'EVAL' ] ) as fin:
        for i, line in enumerate( fin ):
            img, t  = line.split()
            img1    = sentences[ i ].split()[ 0 ]
            if img == img1:
                truths.append( t )
    if not len( truths ):
        print( "Error: no correpondence found between sentences and ground truths" )
        sys.exit()

    return sentences, truths


def pwrite( s, fout=None ):
    """ -----------------------------------------------------------------------------------------------------
    print the string s or write on file if fout is a file descriptor
    ----------------------------------------------------------------------------------------------------- """
    if fout is None:
        print( s )
    else:
        fout.write( s + '\n' )


def print_eval( ref, fout=None ):
    """ -----------------------------------------------------------------------------------------------------
    print the results of the evaluation, on file if fout is a file descriptor
    ----------------------------------------------------------------------------------------------------- """
    tacc    = 0
    ttot    = 0
    for comp in ref.keys():
        acc     = 0
        tot     = 0
        pwrite( comp.center( 40, '-' ), fout=fout )
        truths  = tuple( ref[ comp ].keys() )
        fmt     = 10 * ' ' + len( truths ) * '{:^10}'
        header  = fmt.format( *truths )
        pwrite( header, fout=fout )
        for t in truths:
            s   = '{:^10}'.format( t )
            for p in truths:
                s       += '{:^10d}'.format( ref[ comp ][ t ][ p ] )
                tot     += ref[ comp ][ t ][ p ]
                ttot    += ref[ comp ][ t ][ p ]
                if t == p:
                    acc     += ref[ comp ][ t ][ p ]
                    tacc    += ref[ comp ][ t ][ p ]
            pwrite( s, fout=fout )
        pwrite( 40 * '-', fout=fout )
        if tot:
            pwrite( "accuracy: {:^6.4f}".format( acc / tot ), fout=fout )
            pwrite( 40 * '-' + '\n', fout=fout )
    if ttot:
        pwrite( "total accuracy: {:^6.4f}".format( tacc / ttot ), fout=fout )
    pwrite( 40 * '-' + '\n', fout=fout )
                
            


if __name__ == '__main__':
    args    = get_args()

    if args[ 'PLOT' ] is not None and args[ 'TEXT' ] is None:
        print( "Error: in order to plot SPA you should supply a sentence, using --text" )
        sys.exit()

    if args[ 'EVAL' ] is not None and args[ 'LOAD' ] is None:
        print( "Error: in order to evaluate you should supply a file with sentences, using --load" )
        sys.exit()

    if args[ 'FULL' ] is not None and args[ 'EVAL' ] is None:
        print( "Error: full report is produced in evaluation mode only, you should use --eval" )
        sys.exit()

    if args[ 'METHOD' ] not in kspa_model.methods:
        print( "Error: method {} not implemented".format( args[ 'METHOD' ] ) )
        sys.exit()

    kspa_model.NO_GPU       = False
    kspa_model.n_tsteps     = args[ 'NTSTEPS' ]
    kspa_model.method       = args[ 'METHOD' ]
    kspa_model.batch_pro    = args[ 'BATCH' ]
    kspa_model.verbose      = args[ 'VERBOSE' ]
    if args[ 'PLOT' ] is not None:
        kspa_model.timestep = args[ 'PLOT' ] / kspa_model.n_tsteps
    kspa_model.setup()

    if args[ 'TEXT' ] is not None:
        if args[ 'PLOT' ] is not None:
            kspa_model.evolve_spa( args[ 'TEXT' ] )
            sys.exit()
        res = kspa_model.disambiguate( args[ 'TEXT' ] )
        print( res[ 0 ] + args[ 'SEPARATOR' ] + res[ 1 ] )
        sys.exit()

    if args[ 'LOAD' ] is not None:
        if not os.path.isfile( args[ 'LOAD' ] ):
            print( "Error: file {} not found".format( args[ 'LOAD' ] ) )
            sys.exit()
        fout    = None
        if args[ 'OUTPUT' ] is not None:
            fout    = open( args[ 'OUTPUT' ], 'w' )
        if args[ 'EVAL' ] is None:
            with open( args[ 'LOAD' ] ) as fin:
                for line in fin:
                    res = kspa_model.disambiguate( line )
                    r   = res[ 0 ] + args[ 'SEPARATOR' ] + res[ 1 ]
                    if args[ 'OUTPUT' ] is not None:
                        fout.write( r + '\n' )
                    else:
                        print( r )
        else:
            full    = None
            if args[ 'FULL' ] is not None:
                full    = open( args[ 'FULL' ], 'w' )
            sentences, truths   = read_files()
            class_res           = kspa_model.evaluate( sentences, truths, full=full )
            print_eval( class_res, fout=fout )

        if args[ 'OUTPUT' ] is not None:
            fout.close()
        if args[ 'FULL' ] is not None:
            full.close()
        
                

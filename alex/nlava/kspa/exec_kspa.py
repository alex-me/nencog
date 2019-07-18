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
            default         = 5,
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
            action          = 'store_true',
            dest            = 'VERBOSE',
            help            = "be verbose, display Nengo progress bar"
    )

    return vars( parser.parse_args() )



if __name__ == '__main__':
    args    = get_args()

    if args[ 'METHOD' ] not in kspa_model.methods:
        print( "Error: method {} not implemented".format( args[ 'METHOD' ] ) )
        sys.exit()
    kspa_model.n_tsteps = args[ 'NTSTEPS' ]
    kspa_model.method   = args[ 'METHOD' ]
    kspa_model.verbose  = args[ 'VERBOSE' ]
    kspa_model.setup()

    if args[ 'TEXT' ] is not None:
        res = kspa_model.disambiguate( args[ 'TEXT' ] )
        print( res[ 0 ] + args[ 'SEPARATOR' ] + res[ 1 ] )

    if args[ 'LOAD' ] is not None:
        if not os.path.isfile( args[ 'LOAD' ] ):
            print( "Error: file {} not found".format( args[ 'LOAD' ] ) )
            sys.exit()
        if args[ 'OUTPUT' ] is not None:
            f   = open( args[ 'OUTPUT' ], 'w' )
        with open( args[ 'LOAD' ] ) as fin:
            for line in fin:
                res = kspa_model.disambiguate( line )
                r   = res[ 0 ] + args[ 'SEPARATOR' ] + res[ 1 ]
                if args[ 'OUTPUT' ] is not None:
                    f.write( r + '\n' )
                else:
                    print( r )
        if args[ 'OUTPUT' ] is not None:
            f.close()
        
                

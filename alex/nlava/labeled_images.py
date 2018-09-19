"""
list the images with labels in the json file
"""

import  json
import  os
import  sys

ldir    = "../lava/corpus/"
jfile   = "lava.json"
idir    = "images"
vfield  = "visualFilename"
tfield  = "text"

ls      = os.listdir( os.path.join( ldir, idir ) )
imgs    = [ i.split( '.' )[ 0 ] for i in ls ]
jdata   = json.load( open( os.path.join( ldir, jfile ) ) )
for d in jdata:
    if vfield in d.keys():
        i   = d[ vfield ]
        if not i in imgs:
            print "ERROR: image " + i + " not in the corpus"
            sys.exit( 1 )
        print d[ vfield ] + '\t' + d[ tfield ]

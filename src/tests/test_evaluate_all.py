""" Test the evaluate_all function. """

import argparse
from model import Model
from activations import evaluate_all

def test_evaluate_all( verbose: bool = False ):
    print("# Running test: test_evaluate_all")
    opt = Model("125m", limit=1000)
    if verbose:
        opt.show_details()

    data = evaluate_all( opt, 1e4, verbose=verbose )

    if verbose:
        #print( data )
        print( data.keys() )

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--verbose", "-v", action="store_true", default=False,
        help="Print extra information" )
    args = parser.parse_args()

    test_evaluate_all( verbose=args.verbose )

""" Test the evaluate_all function. """

import argparse
from model import Model
from activations import evaluate, evaluate_all

def test_evaluate( verbose: bool = False ):
    print("# Running test: test_evaluate")
    opt = Model("125m", limit=1000)
    if verbose:
        opt.show_details()

    data_1 = evaluate( opt, 'pile', 1e4, verbose=verbose,
        dataset_texts_to_skip=0 )

    data_2 = evaluate( opt, 'pile', 1e4, verbose=verbose,
        dataset_texts_to_skip=10 )

    keys = data_1.keys()
    assert len(keys) == 10

    for key in keys:
        if key == 'percent':
            continue
        if key == 'token_counts':
            assert data_1[key] is None
            assert data_2[key] is None
            continue
        if verbose:
            print( key, data_1[key], data_2[key] )
        assert data_1[key] != data_2[key]
        assert data_1[key]*0.9 < data_2[key]
        assert data_1[key]*1.1 > data_2[key]

    return True

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

    test_evaluate( verbose=args.verbose )
    test_evaluate_all( verbose=args.verbose )

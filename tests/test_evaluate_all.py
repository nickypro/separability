""" Test the evaluate_all function. """

import argparse
from model import Model
from activations import evaluate, evaluate_all

def test_evaluate( verbose: bool = False ):
    print("# Running test: test_evaluate")
    opt = Model("125m", limit=1000)
    if verbose:
        opt.show_details()

    # We run the a first time, with a small subset of data
    data_1 = evaluate( opt, 'pile', 1e4, verbose=verbose,
        dataset_texts_to_skip=0 )

    # We check that the data is correct
    keys = data_1.keys()
    percent_keys = data_1["percent"].keys()
    assert len(keys) == 10
    assert len(percent_keys) == 4

    # We run the a second time, with a different subset of data
    data_2 = evaluate( opt, 'pile', 1e4, verbose=verbose,
        dataset_texts_to_skip=10 )


    # We check that the output is different, since the input was different,
    # and that the output is similar, since the model is the same, and the text
    # is similar (it's the same dataset, just a different subset)
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
    if verbose:
        print()

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

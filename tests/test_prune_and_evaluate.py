from model import Model
from activations import prune_and_evaluate

def test_prune_and_evaluate( verbose: bool = False ):
    print( "# Running Test: test_prune_and_evaluate" )

    opt = Model("125m", limit=1000)
    data = prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)

    assert data['pile_loss'] < data['code_loss']

    return True

if __name__ == "__main__":
    test_prune_and_evaluate( verbose=True )

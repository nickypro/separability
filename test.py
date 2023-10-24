from separability.model_repos import test_model_repos
from separability import Model
from separability.eval import evaluate, evaluate_all

print("# Running test: test_evaluate_all")
opt = Model("nickypro/tinyllama-15m", limit=1000, dtype="fp32")
data = evaluate_all( opt, 1e3 )
print( data.keys() )

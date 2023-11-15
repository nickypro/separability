from separability.model_repos import test_model_repos
from separability import Model
from separability.eval import evaluate_all, run_evaluation
from separability.texts import infer_dataset_config

print("# Running test: test_evaluate_all")
m = Model("nickypro/tinyllama-15m", limit=1000, dtype="fp32")


print("RUNNING DATASETS")
data = evaluate_all(m, 1e3, datasets=["pile_codeless", "python", "mmlu:high_school_biology", "civil", "wiki"])
print(data["accuracy"])

print("RUNNING TOXICITY")
eval_config = infer_dataset_config("toxicity")
eval_config.generated_text_num_samples = 10
eval_config.verbose = True
out = run_evaluation(m, eval_config)
print(out)

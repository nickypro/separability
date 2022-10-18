from dataclasses import dataclass
from datasets import load_dataset
from evaluate import evaluator

Text2TextEvaluator = evaluator("text2text-generation")

def load_code():
    dataset = load_dataset("codeparrot/codeparrot-clean-valid")
    print( dir(dataset) )
    print( dataset )
    return dataset

if __name__ == "__main__":
    dataset = load_code()
    print( dataset['train'][1]['content'] )




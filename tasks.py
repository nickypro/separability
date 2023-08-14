from datasets import load_dataset, get_dataset_config_names

dataset = load_dataset("tasksource/mmlu", "high_school_computer_science")

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def format_question(datum, include_answer=False):
    s  = "Question:\n"
    s += datum["question"]
    s += "\n\n"
    s += "Choices:"
    for index, choice in enumerate(datum["choices"]):
        s += f"\n{letters[index]}) {choice}"
    s += "\n\n"
    s += "Answer: "
    if include_answer:
        s += letters[datum["answer"]]
    return s

for datum in dataset["test"]:
    print(format_question(datum))

    break

exit()
from tasksource import MultipleChoice


mmlu = MultipleChoice(
    'question',
    choices_list='choices',
    labels='answer',
    splits=['validation','dev','test'],
    dataset_name='tasksource/mmlu',
    config_name="high_school_computer_science",
)

dataset = mmlu.load()

for datum in dataset['test']:
    print(datum)
    break


exit()
from tasksource import MultipleChoice
from datasets import get_dataset_config_names

config_name = get_dataset_config_names("tasksource/mmlu")

for c in ["high_school_computer_science"]:
    mmlu = MultipleChoice(
        'question',
        choices_list='choices',
        labels='answer',
        splits=['validation','dev','test'],
        dataset_name='tasksource/mmlu',
        config_name=c,
    )

    dataset = mmlu.load()
    i = 0
    s = ""

    def generate_question(datum):
        s  = "Question:\n"
        s += datum["question"]
        s += datum


    for datum in dataset['test']:
        print(datum)
        exit()

    print(c, s)


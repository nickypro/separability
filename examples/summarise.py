import json
import openai
from summary_prompt import short_prompt
from tqdm import tqdm

# create a chat completion
def ask_chatgpt(prompt, model="gpt-3.5-turbo"):
    chat_completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
    return chat_completion.choices[0].message.content

def generate_summary(block_text):
    text_input = short_prompt + "\n\n" + "Block:\n" + block_text + "\n\n"
    return ask_chatgpt(text_input)

def generate_outline(text, window_length=256):
    inpt, remainder = text[:window_length], text[window_length:]
    _inputs = []
    _summaries = []
    while len(inpt) > 16:
        _inputs.append(inpt)
        _summaries.append(generate_summary(inpt))
        inpt, remainder = remainder[:window_length], remainder[window_length:]

    return _inputs, _summaries




# load outputs_open_llama_3b_v2.json
with open('outputs_open_llama_3b_v2.json') as json_file:
    generations = json.load(json_file)

# Append outputs to summaries_open_llama_3b_v2.py
with open('summaries_open_llama_3b_v2.py', 'a') as f:
    # load prompt
    num_gen = len(generations)

    # tqdm for loop
    for i in (pbar := tqdm(range(num_gen))):
        gen = generations[i]
        try:
            inputs, summaries = generate_outline(gen["output"])
            s = json.dumps(summaries)
            out_str = f'    ("{gen["input"]}", {s}),\n'
            f.write(out_str)

            # update tqdm string to be input
            tqdm_str = f'"{gen["input"]}"'
            tqdm_str = tqdm_str[:min(len(tqdm_str), 20)]
            tqdm_str = tqdm_str + " " * (20 - len(tqdm_str))
            pbar.set_description(tqdm_str)

        except Exception as e:
            print(e)
            print(f"Skipping {i}: {gen['input']}")

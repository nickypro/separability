from separability import Model
from separability.data_classes import PruningConfig

c = PruningConfig(
    model_repo  = "openlm-research/open_llama_3b",
    token_limit = 2000,
    dtype       = "fp16",
)

opt = Model(c.model_size, limit=c.token_limit, dtype=c._dtype, svd_attn=c.svd_attn)

text_input_acx = """Why Is The Academic Job Market So Weird?
18 MAY 2023

Bret Devereaux [writes here](https://acoup.blog/2023/04/28/collections-academic-ranks-explained-or-what-on-earth-is-an-adjunct/) about the oddities of the academic job market.

His piece is comprehensive, and you should read it, but short version: professors are split into tenure-track (30%, good pay and benefits) and adjunct (50%, bad pay and benefits). Another 20% are “teaching-track”, somewhere in between.

Everyone wants a tenure-track job. But colleges hiring new tenure-track faculty prefer newly-minted PhDs to even veteran teaching-trackers or adjuncts. And even if they do hire a veteran teaching-tracker or adjunct, it’s practically never one of their own. If a teaching-tracker or adjunct makes a breakthrough, they apply for a tenure-track job somewhere else. Devereaux describes this as “a hiring system where experience manifestly hurts applicants” and displays this graph:


Number of professors hired for tenure-track positions by how long it’s been since the candidate has gotten their PhD.
He focuses on the moral question: is this good (no), and how can it be stopped (activism). I appreciate his commentary but I found myself wondering about the economic question: why did the system end up like this?

"""

text_input_counting = """1 2 3 4 5 6 7 8 9 10
11 12 13 14 15 16 17 18 19 20
21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 36 37 38 39 40
41 42 43 44 45 46 47 48 49 50

1 2 3 4 5 6 7 8 9 10
11 12 13 14 15 16 17 18 19 20
21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 36 37 38 39 40
41 42 43 44 45 46 47 48 49 50

1 2 3 4 5 6 7 8 9 10
11"""

text_input_poem = """Sonnet 18: Shall I compare thee to a summer’s day?
BY WILLIAM SHAKESPEARE
Shall I compare thee to a summer’s day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer’s lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature’s changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow’st;
Nor"""

text_input = text_input_poem
text_input = "Hello World"

for i in range(10):
    inpt, output = opt.generate(text_input, num=1000, do_sample=False, temperature=0.1)
    print(f'"{"".join(inpt)}"')
    print(f'"{"".join(output)}"')



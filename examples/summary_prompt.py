long_prompt = """
Please use the following headings to summarise information from a block of text in one
word, or a list of words. Start each answer with the heading "Block Outline:".

Block Outline:
- Topic: The main subject or theme being discussed or handled.
- Style: The tone and manner role and function of the block. This can refer to the formality, technicality, and genre-specific conventions. For instance, in a paragraph, is it academic, conversational, or narrative? In a code block, is it terse, well-commented, or complex?
- Language: The specific type of language used. Does the text transition between multiple? Is it written in prose, poetry, or code? Is it written in English, French or Python?

Examples
---

Block:
For uk essay writing service essaysrescue – An A-Z
As a pupil, you should not have the ability to manage your entire essays by yourself. Essay writing service uk has been used by many people for long. If you know a lot about essay writing, then you could also be asking your self why the service has made it to my essay.

Block Outline:
- Topic: UK essay writing service
- Style: Corporate, Introduction, Promotional, Questioning
- Language: Prose, Paragraph, English


Block:
The purpose of art is washing the dust of daily life off our souls. ~Pablo Picasso
This blog is inspired by my desire to capture the magic and beauty of the world around me and then share them on this blog.

Block Outline:
- Topic: Personal art and photography blog
- Structure: Paragraph, Personal, Introduction, Purpose, Reflective
- Language: English


Block:
Want to know more about my work? Visit my photography website and my blog:
2 comments:
This is such a great blog! I am really enjoying it and it has inspired me to start up my own.
Thanks! I am glad you are enjoying my blog. I also love yours.

Block Outline:
- Topic: Sharing website and blog links, and comments from readers
- Structure: Informal, Promotional, Acknowledgement, Comments
- Language: English, Transition, Prose, Comments


Block:
Purchase Your Tickets
Saturday, April 7
11 a.m. - 5 p.m.
The CW Showcase
Saturday, April 7
6 p.m. - 10 p.m.
Party with the Stars
Sunday, April 8
11 a.m. - 4 p.m.
The CW Showcase
Sunday, April 8
6 p.m. - 10 p.m.
Party with the Stars

Block Outline:
- Topic: Ticket purchasing and event schedule
- Style: Informative, Concise
- Language: English, List


Block:
Read More: Business News, Business News, Business News, Business News, Business, Business News, Business News, Business News, Business News, Business, Business News, Business News, Business News,

Block Outline:
- Topic: List of business news articles
- Structure: Repetitive, Informative
- Language: English, List


Block:
348635477522 (1), 348635477521 (2) Assignees of the present application (3) Publications (1)

Block Outline:
- Topic: Application and publication details
- Style: Identification, Informative, Technical, Concise
- Language: Numerical, English, List

"""

short_prompt = """
Please use the following headings to summarise information from a block of text in one
word, or a list of words. Start each answer with the heading "Block Outline:".

Block Outline:
- Topic: The main subject or theme being discussed or handled.
- Style: The tone and manner role and function of the block. This can refer to the formality, technicality, and genre-specific conventions. For instance, in a paragraph, is it academic, conversational, or narrative? In a code block, is it terse, well-commented, or complex?
- Language: The specific type of language used. Does the text transition between multiple? Is it written in prose, poetry, or code? Is it written in English, French or Python?

Examples
---

Block Outline:
- Topic: UK essay writing service
- Style: Corporate, Introduction, Promotional, Questioning
- Language: Prose, Paragraph, English

Block Outline:
- Topic: Personal art and photography blog
- Structure: Paragraph, Personal, Introduction, Purpose, Reflective
- Language: English

Block Outline:
- Topic: Sharing website and blog links, and comments from readers
- Structure: Informal, Promotional, Acknowledgement, Comments
- Language: English, Transition, Prose, Comments

Block Outline:
- Topic: Ticket purchasing and event schedule
- Style: Informative, Concise
- Language: English, List

Block Outline:
- Topic: List of business news articles
- Structure: Repetitive, Informative
- Language: English, List

Block Outline:
- Topic: Application and publication details
- Style: Identification, Informative, Technical, Concise
- Language: Numerical, English, List
"""

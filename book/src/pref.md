# Preface

As a compiler writer for domain specific cloud languages (Terraform HCL),
I became interested in compiler implementations for domain specific tensor languages
such as PyTorch 2 after the software 3.0 unlock of natural language programming from large language models such as ChatGPT.
However, I became frustrated with the *non-constructiveness* and *disjointedness* of
my learning experience in the discipline of machine learning systems,
as I preferred a style of pedagody similar to the introductory computer science canon created by Schemers
taught to first years at Waterloo, which consists of two books that secretly masquerade as one
— [SICP](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html) and it's [dual](https://cs.brown.edu/~sk/Publications/Papers/Published/fffk-htdp-vs-sicp-journal/paper.pdf), [HTDP](https://htdp.org/), teach programming
and programming languages by taking readers through an unbroken logical sequence in a [flânnerie](https://cs.uwaterloo.ca/~plragde/flaneries/)-like style. The recent addition of [DCIC](https://dcic-world.org/), spawning from it's phylogenetic cousin [PAPL](https://papl.cs.brown.edu/2020/), was created to adjust the curriculum to the recent [shift in data science](https://cs.brown.edu/~sk/Publications/Papers/Published/kf-data-centric/paper.pdf) by focusing the *tabular/table* data structure.
Given the generality of the attention mechanism autoregressively predicing the next-token on internet data,
this book follows suit (*aspirationally* titled SITP),
and focuses on the stochastically continuous computational mathematics
and low level systems programming necessary for training deep neural networks.

If you are more experienced, you may benefit in jumping straight to part three of the book
which develops a "graph mode" fusion compiler and inference engine with tinygrad's RISCy IR,
borrowing ideas from ThunderKitten's tile registers, MegaKernels, and Halide/TVM schedules.
Beyond helping those like myself interested in the systems of deep learning,
developing the low level performance primitives of deep neural networks will shed light on the open research question of
how domain specific tensor languages of deep learning frameworks can best support the development and compilation of accelerated kernels for novel network architectures (inductive biases) beyond the attention mechanism of transformers.

One question you may ask is why spend the time and effort
to study the following contents in the form of a book when you can prompt your favorite large language model yourself.
The preface of a introductory book on artificial intelligence would be remiss without addressing such a
substantial question in the age of software 3.0.
We will answer first with why people should continue to read books, and second why people should continue to learn programming.

1. With respect to the media of the contents being presented a *book-format*, it's because the library's *book*, the internet's *article*, and the journal's *paper*, continue to provide *complementary value* to the large language model's *answer*.
First, although not totally binary and moreso a spectrum,
there is a pull-like nature when working with large language models in that it's the prompters responsibility
to direct the model into the intended latent space of understanding.
As a result, they a more well suited for knowledge needed just-in-time.
In contrast, books provide more of a push-like nature which provide this set table of contents up front,
revealing many unknown unknowns to the uninitiated beginner which have a more *normative, platonic* nature to them
in that this set syllabus *should* be learned. In other words, books continue to form *canon*. As a result, they are more well suited for knowledge needed ahead-of-time.
platonic.
canon.

2. With respect to the contents being presented themself of *programming software 2.0*,
- abstractions are leaky
- the current way ai is being programmed, nothing prevents understanding (holding state in your head)
- interface with the llm
- horse vs engine. or limbic system and neocortex?
- jevon's paradox
- but as always, the market will decide
- engineer-> research engineer (entrepeneur.) agency: instantiate something ood.

<!-- The SITP book takes readers from training models to developing their own deep learning frameworks.
This book has been creatively handwritten for humans, and for now, achieves such a goal better
than prompting a state of the art LLM to *"write me the SICP for software 2.0"*
— the number of books that comprise this style is not enough for the sample efficiency of today's SOTA LLMs.

understanding.
Agents, what a system should platonically do.
link to tinygrad devlog. -->

If you empathize with some of my motivations, you may benefit from the book too[^0].</br>
Good luck on your journey.</br>
Are you ready to begin?</br>

---
[^0]: *And if not, I hope this book poses as a good counterexample for what you have in mind.*
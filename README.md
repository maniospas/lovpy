# Lovpy
Logic verification library for Python data exchanges.

**Dependencies:** `networkx, matplotlib, pandas, pywin32, sklearn, pygraphviz`, `stellargraph` (optional)<br>
**Authors:** Dimitrios Karageorgiou, Emmanouil Krasanakis<br>
**Contact:** soulrain@outlook.com<br>
**Licence:** Apache 2.0

# :rocket: Features
- Recognize problems before side-effects
- No false alarms
- No code modification
- Easy debugging (shows last correct line of code)

# :zap: Quickstart
Lovpy can be installed with the command line instruction 
`pip install lovpy`.

First, let us define a generic-purpose list of specification that monitor
threading. This follows standard Gherkin syntax and refers to code
data as actors. Each text expression is considered one predicate
and expressions in the scope `$...$` are evaluated dynamically on
running code.

```shell
SCENARIO:
    WHEN returned by allocate_lock
    THEN NOT locked $threading.get_ident()$

SCENARIO:
    WHEN call acquire
    THEN SHOULD NOT locked_$threading.get_ident()$
    AND locked $threading.get_ident()$

SCENARIO:
    GIVEN locked_$threading.get_ident()$
    WHEN call release
    THEN PRINT released by [METHOD]
    AND NOT locked $threading.get_ident()$
```

Place the above specification in a `specifications.gherkin` file within
your project. You can have any file name and any number of files.

Then create normal python threading code, for instance in a 
`script.py` file. Calling this with online verification 
of the specification can be done per 

```
python -m lovpy script.py
```

If a violation is detected, an appropriate exception is raised,
like the one below.
If applicable, the last provably correct line of code is reported, 
so you need only check code after that point to fix the bug.

```shell
Traceback (most recent call last):
  File "examples\invalid_thread_test.py", line 43, in <module>
  File "examples\invalid_thread_test.py", line 33, in get_both_parts <-- LAST CORRECT LINE, line 31
    second = get_second_part()
  File "examples\invalid_thread_test.py", line 10, in get_first_part
    try:
lovpy.exceptions.PropertyNotHoldsException: A property found not to hold:
        WHEN call acquire THEN SHOULD NOT locked_$threading.get_ident()$ AND locked_$threading.get_ident()$

```

## :link: Material
* [lovpyignore](documentation/ignore.md)
* [GNN verification](documentation/GNN.md)
* [evaluate the system](documentation/evaluation.md)
* [customize run](documentation/parameters.md)

## :computer: Presentation
<iframe src="//www.slideshare.net/slideshow/embed_code/key/4bcbUl4VFDniny" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe>

## :notebook: Citation
To cite `lovpy` please refer to the following diploma thesis:
```
@misc{karageorgiou2021lovpy,
      title={Python metaprogramming in linear time language for automated runtime verification with graph neural networks}, 
      author={Dimitrios Karageorgiou},
      year={2021},
      archivePrefix={https://ikee.lib.auth.gr/record/},
      eprint={335121},
}
```
# Lovpy
#### A simple-to-use, yet powerful, logic verification library for Python.

## Description
Lovpy is a python library for performing logic verification at runtime. Logic verification defines
a broad scientific area ranging from runtime verification to enforcement of good development practices, by
verifying the expected behavior of the program. Lovpy utilizes Gherkin, the popular, simple and
intuitive specifications language of Cucumber, for specifying that expected behavior. Using an innovative
verification engine, assisted by the power of deep graph neural networks, it is able to 
detect many kinds of violations. Lovpy not only reports the line of code where a 
violation happened, but is also able to report the last line which was provably correct. Each reported violation 
is based on strong mathematical proofs, so a guarantee is provided for zero false reports.
Finally, respecting the moto *simplicity above all*, using Lovpy requires no code changes at all. 

### Features
- No code modifications required to enable verification.
- Specifications in an easy-to-learn and intuitive language (Gherkin).
- Never reports a violation that does not exist (0% false-negatives).
- Reports violations before they happen (prevent side-effects).
- Reports the line of code where violation detected.
- Reports the last provably correct line of code (use for debugging).

A quick scientific presentation of lovpy is [available here](https://www.slideshare.net/isselgroup/python-metaprogramming-in-linear-time-language-for-automated-runtime-verification-with-graph-neural-networks)!

A thorough scientific presentation is also [available here](https://ikee.lib.auth.gr/record/335121/?ln=en) (currently in Greek, due to requirements of the university).

## Quick Start
Lovpy is available at PyPI and can be installed as following:

- `pip install lovpy`

Then, in order to verify that a python program conforms to a set of specifications written in Gherkin:
1. Place the `.gherkin` specifications file under current working directory.
2. Run any script like: `py -m lovpy <script.py> <args>`

If a violation is detected, an appropriate exception is raised. Also, if applicable, the last provably correct line of code is reported, requiring from developers to only check the intermediate code in order to fix the bug.
![Exception raised when detected a violation.](https://user-images.githubusercontent.com/33910136/148264808-37ad60c9-63d0-4cf3-a5a6-bbeb5c776b4a.png)

## Verification engines
Internally, lovpy converts everything into theorems to be mathematically proved. Proof is performed by a novel theorem proving engine, based on *temporal graphs*. Currently, many different verification engines co-exist:

- **Basic**: Utilizes heuristic rules in order to prove violations. This is the fastest running engine, able to prove a great amount of violations, requiring no trained models at all.
- **GNN**: Utilizes deep graph neural network models in order to prove violations.
- **MLP**: Utlizes simple neural models based on multi-layer perceptrons for the proving process. It is mostly used as a reference baseline for the capabality of the system.
- **Hybrid**: The most powerfull verification engine currently contained in lovpy. Utilizes both GNN models and heuristic rules in order to prove violations.

In order to use the three neural verification engines, `tensorflow` and `stellargraph` packages are required. By default, lovpy does not install them, so only the basic engine is immediately available. In order to install them, use the following `pip` command:

- `pip install tensorflow stellargraph`

## Models training

In order to fully utilize the power of neural provers, corresponding models should be trained beforehand. In order te perform model training, the following command can be utilized:
- `py -m lovpy -t <all | simple | gnn>`
It trains graph neural networks based models when `gnn` argument is provided and multi-layer perceptron based ones when `simple` is provided. In order to train both, just provide the `all` argument.

Location of models can be defined by the user through setting `LOVPY_MODELS_DIR = <dir>` environmental variable. It defaults to a directory named `.lovpy` under system's home directory.

It is also possible to programmatically trigger training of models if they do not exist. This is mostly useful when integrating lovpy into 3rd party libraries.
```
from lovpy import load_or_train()
load_or_train()
```

## Exclude source files from verification
Lovpy allows control of which python source files to be verified through the use of `.lovpyignore` files. Inspired by gitignore files, they are used in quite a similar way. All you have to do is to place a file named `.lovpyignore` under any directory of your project and inside it define files or folders to be excluded. Paths are resolved relatively to the location of `.lovpyignore` file. `*` and `**` can be used as wildcards in they same way they are used in `glob` module. An example `.lovpyignore` file is presented below:
```
source
tests
venv
bin/*.py
```

## Evaluation

Evaluation of the library can be performed either against included code examples or against synthetically generated theorems using the following command:
- `py -m lovpy -e <examples | synthetics>`

## Supported Environmental Variables:
- `LOVPY_ENGINE = BASIC | MLP | GNN | HYBRID` : Explicitly enables a specific verification engine.  
- `LOVPY_DISABLE_GPU = 0 | 1` : When set to `1` disables GPU usage by tensorflow.
- `LOVPY_SESSION_NAME = <name>` : Sets a custom name for current session.
- `LOVPY_TEMP_DIR = <dir>` : Directory where lovpy will store all data and reports of a session.
- `LOVPY_MODELS_DIR = <dir>` : Directory which lovpy will use for storing and loading models. 
- `LOVPY_DEV_MODE = 0 | 1` : When set to `1` enables development mode.

## Complete reference of supported Gherkin commands
[TODO]

## License:
This project is licensed under Apache License 2.0. A copy of this license is contained in current project under `LICENSE` file. It applies to all files in this project whether or not it is stated in them.

Copyright 2021 | Dimitrios S. Karageorgiou

# Lovpy
#### A simple-to-use, yet powerful, logic verification library for Python.

## Description:
Lovpy is a python library for performing logic verification at runtime. Logic verification covers
the wide area of software runtime verification and enforcement of good development practices, by
verifying the expected behavior of the program. Lovpy utilizes Gherkin, the popular, simple and
intuitive language of Cucumber, for specifying that expected behavior. Using an innovative
verification engine, assisted by the power of deep graph neural networks, it is able to 
detect violations of specifications. Not only it is able to detect the line in code where a 
violation happens, but can also report the last line which was provably correct. Each report 
is based on strong mathematical proofs, so guarantee is provided for no false reports.
Finally, respecting the moto *simplicity above all*, using Lovpy requires no code changes at all. 

# Installation
Lovpy is available under PyPI and can be installed as following:

`pip install lovpy`

By default, lovpy does not install tensorflow, so only the basic engine is available for verification. In order to utilize neural engines, `tensorflow` and `stellargraph` should be installed manually, as following:

`pip install tensorflow stellargraph`

## Basic usage:
### Start verification:
`python -m lovpy <script.py>`

An alternative way to start validation programmatically is the addition of a single python line:

`import lovpy`

### Train models:
`python -m lovpy -t <all | simple | gnn>`

### Evaluate:
`python -m lovpy -e <examples | synthetics>`

### Supported Environmental Variables:
- `LOVPY_ENGINE = BASIC | MLP | GNN | HYBRID` : Explicitly enables a specific verification engine.  
- `LOVPY_DISABLE_GPU = 0 | 1` : When set to `1` disables GPU usage by tensorflow.
- `LOVPY_SESSION_NAME = <name>` : Sets a custom name for current session.
- `LOVPY_TEMP_DIR = <dir>` : Directory where lovpy will store all data and reports of a session.
- `LOVPY_MODELS_DIR = <dir>` : Directory which lovpy will use for storing and loading models. 
- `LOVPY_DEV_MODE = 0 | 1` : When set to `1` enables development mode.

## License:
This project is licensed under Apache License 2.0. A copy of this license is contained in current project under `LICENSE` file. It applies to all files in this project whether or not it is stated in them.

Copyright 2021 | Dimitrios S. Karageorgiou
#Logipy
## An easy-to-use yet powerful logic verification library for Python.

Developed by *Dimitrios Karageorgiou*,\
during diploma thesis on *Automated Proving using Graph Neural Networks for Logic Verification of Code at Runtime*,\
*Electrical and Computers Engineering Department,*\
*Aristotle University Of Thessaloniki, Greece,*\
*2020-2021.*

## Description:
TODO

# Installation
TODO

## Basic usage:
### Start validation:
`python -m logipy <script.py>`

An alternative way to start validation programmatically is the addition of a single python line:\
`import logipy`

### Train models:
`python -m logipy -t <all | simple | gnn>`

### Evaluate:
`python -m logipy -e <examples | synthetics>`

### Supported Environmental Variables:
- `LOGIPY_DISABLE_GPU = 0 | 1` : When set to `1` disables GPU usage by tensorflow.
- `LOGIPY_SESSION_NAME = <name>` : Sets a custom name for current session.
- `LOGIPY_TEMP_DIR = <dir>` : Sets directory where logipy will store all data and reports of a session.
- `LOGIPY_DEV_MODE = 0 | 1` : When set to `1` enables development mode.

## License:

This project is licensed under Mozilla Public License Version 2.0. A copy of this license is contained in current project. It applies to all files in this project whether or not it is stated in them.


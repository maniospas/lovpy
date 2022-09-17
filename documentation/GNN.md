# GNN verification
Internally, lovpy converts everything into theorems to be mathematically proved. Proof is performed by a novel theorem proving engine, based on *temporal graphs*. Currently, many different verification engines co-exist:

- **Basic**: Utilizes heuristic rules in order to prove violations. This is the fastest running engine, able to prove a great amount of violations, requiring no trained models at all.
- **GNN**: Utilizes deep graph neural network models in order to prove violations.
- **MLP**: Utilizes simple neural models based on multi-layer perceptrons for the proving process. It is mostly used as a reference baseline for the capabality of the system.
- **Hybrid**: The most powerful verification engine currently contained in lovpy. Utilizes both GNN models and heuristic rules in order to prove violations.

In order to use the three neural verification engines, `tensorflow` and `stellargraph` packages are required. By default, lovpy does not install them, so only the basic engine is immediately available. In order to install them, use the following `pip` command:

- `pip install tensorflow stellargraph`

## Model training
In order to fully utilize the power of neural provers, corresponding models should be trained beforehand. In order te perform model training, the following command can be utilized:
- `py -m lovpy -t <all | simple | gnn>`
It trains graph neural networks based models when `gnn` argument is provided and multi-layer perceptron based ones when `simple` is provided. In order to train both, just provide the `all` argument.

Location of models can be defined by the user through setting `LOVPY_MODELS_DIR = <dir>` environmental variable. It defaults to a directory named `.lovpy` under system's home directory.

It is also possible to programmatically trigger training of models if they do not exist. This is mostly useful when integrating lovpy into 3rd party libraries.
```
from lovpy import load_or_train()
load_or_train()
```
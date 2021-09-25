import runpy
import sys
from sys import argv

from .models.train_model import train_models
from .evaluation.evaluate_on_synthetics import evaluate as eval_on_synthetics
from .evaluation.evaluate_on_examples import evaluate_proving_methods as eval_on_examples
from .config import VERSION


def main():
    if len(argv) > 1 and argv[1].endswith(".py"):
        print("-" * 80)
        print(f"Running {argv[1]} under logipy's verification.")
        print("-" * 80)
        sys.argv = sys.argv[1:]
        runpy.run_path(sys.argv[0], run_name="__main__")

    elif len(argv) > 1 and (argv[1] == "--train" or argv[1] == "-t"):
        if len(argv) > 2 and argv[2] == "simple":
            train_models(arch="simple")
        elif len(argv) > 2 and argv[2] == "gnn":
            train_models(arch="gnn")
        else:
            train_models()

    elif len(argv) > 1 and (argv[1] == "--eval" or argv[1] == "-e"):
        if len(argv) > 2 and argv[2] == "examples":
            eval_on_examples()
        elif len(argv) > 2 and argv[2] == "synthetics":
            eval_on_synthetics()

    elif len(argv) > 1 and (argv[1] == "--version" or argv[1] == "-v"):
        print(f"Logipy version: {VERSION}")

    else:
        print("Usage: python -m logipy <script.py>|((-t|--train) [simple|gnn])")
        print("")
        print("Arguments:")
        print("\t-t | --train : Trains available all neural architectures. If one of")
        print("\t               the following modifiers are given, trains only selected")
        print("\t               architecture.")
        print("\t\tsimple : Trains only the simple neural architecture.")
        print("\t\tgnn : Trains only the gnn-based neural architecture.")
        print("\t-e | --eval : Evaluates installed proving systems.")
        print("\t\texamples : Evaluation is performed on code snippets.")
        print("\t\tsynthetics : Evaluation is performed on synthetic samples.")
        print("\t-h | --help : Displays this message.")
        print("\t-v | --version : Displays logipy's version.")


main()

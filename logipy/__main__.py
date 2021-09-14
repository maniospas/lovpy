from sys import argv

from logipy.models.train_model import train_models
from .config import VERSION


def main():
    if len(argv) > 1 and argv[1].endswith(".py"):
        # TODO: Run script from command line.
        print("Script running not implemented yet.")

    elif len(argv) > 1 and (argv[1] == "--train" or argv[1] == "-t"):
        if len(argv) > 2 and argv[2] == "simple":
            train_models(arch="simple")
        elif len(argv) > 2 and argv[2] == "gnn":
            train_models(arch="gnn")
        else:
            train_models()

    elif len(argv) > 1 and (argv[1] == "--eval" or argv[1] == "-e"):
        if len(argv) > 2 and argv[2] == "examples":
            # TODO: Evaluate examples.
            print("Examples evaluation not implemented yet.")
        elif len(argv) > 2 and argv[2] == "synthetics":
            # TODO: Evaluate on synthetic samples.
            print("Synthetic samples evaluation not implemented yet.")

    elif len(argv) > 1 and (argv[1] == "--version" or argv[1] == "-v"):
        print(f"Logipy version: {VERSION}")

    else:
        print("Usage: python -m logipy <script.py>|((-t|--train) [simple|gnn])")
        print("")
        print("Arguments:")
        print("\t-t | --train : Trains all neural architectures available. If one of")
        print("\t               the following modifiers are given, trains only selected")
        print("\t               architecture.")
        print("\t\tsimple : Trains only the simple neural architecture.")
        print("\t\tgnn : Trains only the gnn-based neural architecture.")
        print("\t-h | --help : Displays this message.")
        print("\t-v | --version : Displays logipy's version.")


main()

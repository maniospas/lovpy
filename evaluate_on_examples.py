import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from contextlib import contextmanager
import multiprocessing
import threading
import _thread

import logipy
from logipy.exceptions import PropertyNotHoldsException
from pathlib import Path


EXAMPLES_DIR = "./examples"


class TimeoutException(Exception):
    pass


def evaluate_proving_methods():
    valid_script_paths = list(Path(EXAMPLES_DIR).glob("valid_*.py"))
    invalid_script_paths = list(Path(EXAMPLES_DIR).glob("invalid_*.py"))

    print("-" * 64)
    print("Testing {} scripts:".format(len(valid_script_paths)+len(invalid_script_paths)))
    print("\t-{} scripts are valid.".format(len(valid_script_paths)))
    print("\t-{} scripts contain a bug.".format((len(invalid_script_paths))))

    # print("-" * 64)
    # print("Evaluating deterministic prover.")
    # print("-" * 64)
    # logipy.config.set_theorem_selector(logipy.config.TheoremSelector.DETERMINISTIC)
    # evaluate_on_examples(valid_script_paths, invalid_script_paths)
    #
    # print("-" * 64)
    # print("Evaluating fully-connected NN based prover.")
    # print("-" * 64)
    # logipy.config.set_theorem_selector(logipy.config.TheoremSelector.SIMPLE_NN)
    # evaluate_on_examples(valid_script_paths, invalid_script_paths)

    print("-" * 64)
    print("Evaluating DGCNN based prover.")
    print("-" * 64)
    logipy.config.set_theorem_selector(logipy.config.TheoremSelector.DGCNN)
    evaluate_on_examples(valid_script_paths, invalid_script_paths)


def evaluate_on_examples(valid_script_paths, invalid_script_paths):
    valid_to_valid = []  # TP
    valid_to_invalid = []  # FP
    invalid_to_valid = []  # FN
    invalid_to_invalid = []  # TN
    for p in valid_script_paths:
        print("\t\tEvaluating {}".format(p))
        if evaluate_script(p):
            valid_to_valid.append(p)
        else:
            valid_to_invalid.append(p)
    for p in invalid_script_paths:
        print("\t\tEvaluating {}".format(p))
        if evaluate_script(p, timeout=60.):
            invalid_to_valid.append(p)
        else:
            invalid_to_invalid.append(p)

    print("-" * 64)
    print("\t-{} out of {} valid scripts evaluated wrong.".format(
        len(valid_to_invalid), len(valid_script_paths)))
    for fp in valid_to_invalid:
        print("\t\t{}".format(str(fp)))
    print("\t-{} out of {} invalid scripts evaluated wrong.".format(
        len(invalid_to_valid), len(invalid_script_paths)))
    for fn in invalid_to_valid:
        print("\t\t{}".format(str(fn)))


def evaluate_script(path, timeout=None):
    with path.open("r") as f:
        script = f.read()

    is_valid = True

    try:
        if timeout:
            with time_limit(timeout):  # Use timeout to escape deadlocks.
                exec(script, {})
        else:
            exec(script, {})
    except PropertyNotHoldsException:
        is_valid = False
    except TimeoutException:
        pass

    return is_valid


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


if __name__ == "__main__":
    evaluate_proving_methods()
